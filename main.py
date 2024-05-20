import numpy as np
import torch
import time
import argparse
import model.EMBSFormer as EMBSFormer
import tasks.PreTask as Task
import utils.data.DataSet as Data
import pytorch_lightning as pl
from model_config import EMBSFormer_config as ast_config
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.mail.email_funtion as ef




FIGURE_SAVE_PATH = "./figures"
EMAIL_CONF = "./utils/mail/mail_info.conf"
SaAVE_MODEL = "./model_dict/"

DATA_PATH = {
    "PEMS04": {"data": "./data/PEMS/PEMS04/PEMS04.npz", "adj": "./data/PEMS/PEMS04/PEMS04.csv", "tem_speed": "./data/PEMS/PEMS04/TEMP_PEMS04.csv", "nodes_num": 307, "minute_offset":0},
    "PEMS07": {"data": "./data/PEMS/PEMS07/PEMS07.npz", "adj": "./data/PEMS/PEMS07/PEMS07.csv", "tem_speed": "./data/PEMS/PEMS07/TEMP_PEMS07.csv", "nodes_num": 883, "minute_offset":0},
    "PEMS08": {"data": "./data/PEMS/PEMS08/PEMS08.npz", "adj": "./data/PEMS/PEMS08/PEMS08.csv", "tem_speed": "./data/PEMS/PEMS08/TEMP_PEMS08.csv", "nodes_num": 170, "minute_offset":0}
}

CONFIG_PATH = {
    "EMBSFormer": {
                "PEMS04": "./configurations/EMBSFormer/PEMS04.conf",
               "PEMS07": "./configurations/EMBSFormer/PEMS07.conf",
               "PEMS08": "./configurations/EMBSFormer/PEMS08.conf"
    }
}


def get_exit_model(args, dm):
    device = torch.device("cpu")
    if args.gpus > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    backbones = ast_config.get_model_config(args.config_path, DATA_PATH[args.data_name]["adj"], device)
    dict = {
        "seq_len": int(dm.config["Data"]["seq_len"]),
        "pre_len": int(dm.config["Data"]["pre_len"]),
        "nodes_num": int(dm.config["Data"]["nodes_num"]),
        "feature_dim": int(dm.config['Data']['feature_dim']),
        "embedded_dim": int(dm.config['Training']['embedded_dim']),
        "backbone_list": backbones,
        "heads": int(dm.config['Training']['heads']),
        "num_of_days": int(dm.config['Training']['num_of_days']),
        "num_of_weeks": int(dm.config['Training']['num_of_weeks']),
        "add_time_in_day": dm.config.getboolean('Data', 'add_time_in_day'),
        "add_day_in_week": dm.config.getboolean('Data', 'add_day_in_week'),
        "droupout": float(dm.config['Training']['droupout'])
    }
    model = Task.PreSeqTask.load_from_checkpoint(args.model_path, dict).to(device)
    return model

def get_model(args, dm):
    device = torch.device("cpu")
    if args.gpus>0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.model_name == "EMBSFormer":
        import model_config.EMBSFormer_config as embsformer_config
        backbones = embsformer_config.get_model_config(CONFIG_PATH[args.model_name][args.data_name],
                                                       DATA_PATH[args.data_name]["adj"],
                                                       device)
        model = EMBSFormer.EMBSFormer(int(dm.config['Data']['seq_len']),
                                      int(dm.config['Data']['pre_len']),
                                      DATA_PATH[args.data_name]["nodes_num"],
                                      int(dm.config['Data']['feature_dim']),
                                      int(dm.config['Training']['embedded_dim']),
                                      backbones,
                                      int(dm.config['Training']['heads']),
                                      eval(dm.config['Training']['cycle_matrix_config']),
                                      dm.config.getboolean('Data', 'add_time_in_day'),
                                      dm.config.getboolean('Data', 'add_day_in_week'),
                                      dm.config.getboolean('Data', 'add_holiday'),
                                      float(dm.config['Training']['droupout']))

    return model


def main_task_train(args):
    dm = Data.SpatioTemtoralSeqDataModule(DATA_PATH[args.data_name]["data"],
                                          DATA_PATH[args.data_name]["tem_speed"],
                                          DATA_PATH[args.data_name]["adj"],
                                          DATA_PATH[args.data_name]["minute_offset"],
                                          args.data_name,
                                          args.batch_size,
                                          CONFIG_PATH["EMBSFormer"][args.data_name],
                                          split_ratio=0.7
                                          )
    model = get_model(args, dm)
    task = Task.PreSeqTask(model, data_max_val=dm.stats)
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    trainer = None
    if args.model_path != "":
        trainer = pl.Trainer(resume_from_checkpoint=args.model_path)
        trainer = trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
        task = get_exit_model(args, dm)
    else:
        trainer = pl.Trainer.from_argparse_args(args, auto_select_gpus=True, callbacks=[checkpoint_callback])
    trainer.fit(task, dm)
    # res = trainer.validate(datamodule=dm)
    # test model
    best_model = Task.PreSeqTask.load_from_checkpoint(checkpoint_callback.best_model_path, gpus=1)
    if args.gpus > 0:
        best_model = best_model.to('cuda')
    test_res = trainer.test(model=best_model, dataloaders=dm.test_dataloader())
    # test_res = test_for_train_model(checkpoint_callback, trainer)
    # test_res = trainer.test()
    print("-------test res-----")
    print(test_res)
    ex_res = task.metrics
    res = {"epochs": args.max_epochs, "result": ex_res}
    img_name = task.plot_show(args.model_name, FIGURE_SAVE_PATH)
    if args.pre_curve_figure == True:
        task.plot_prediction(args.model_name+"_pre_curve", FIGURE_SAVE_PATH)
    torch.save(task.model.state_dict(), SaAVE_MODEL + args.model_name + "_" + str(args.max_epochs) + "_" + str('%.2f'%ex_res["MAE"]) + ".pth")
    return res, img_name

# def test_for_train_model(checkpoint_callback, trainer):
#     save_paths = checkpoint_callback.best_k_models_paths
#     res = {}
#     for i in range(len(save_paths)):
#         best_model = Task.PreSeqTask.load_from_checkpoint(save_paths[i])
#         if args.gpus > 0:
#             best_model = best_model.to('cuda')
#         test_res = trainer.test(model=best_model, datamodule=dm)
#         test_res['model_path'] = save_paths[i]
#         if i==1:
#             res = test_res
#         else:
#             if res['MAE'] > test_res['MAE']:
#                 res = test_res
#     return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_name", type=str, help="The name of dataset", choices=("PEMS04", "PEMS07", "PEMS08", "sz-taxi", "los-taxi", "METR-LA"), default="PEMS08")
    parser.add_argument("--model_name",
                        type=str,
                        help="The name of model",
                        choices=("EMBSFormer"),
                        default="EMBSFormer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--send_email", type=bool, default=False)
    parser.add_argument("--pre_curve_figure", type=bool, default=False)

    parser = getattr(Task, "PreSeqTask").add_task_specific_arguments(parser)
    args = parser.parse_args()

    start_time = time.time()
    res, img_name = main_task_train(args)
    end_time = time.time()
    use_time = float(end_time - start_time)
    if args.send_email:
        from utils.mail import email_funtion as ef
        txt = args.model_name + "模型训练用时：" + str(use_time) + "\n" + str(res)
        img_path = "./figures/" + img_name
        ef.send_email(txt, "*** " + args.model_name + " model train result ***", EMAIL_CONF, img_path)
