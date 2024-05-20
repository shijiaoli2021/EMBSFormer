import configparser
import smtplib
# 负责构造文本
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# 负责将多个对象集合起来
from email.mime.multipart import MIMEMultipart
from email.header import Header


EMAIL_CONF = "./mail_info.conf"
'''
Email the result, you can send train result as txt and images to your email 
'''
def send_email(txt, theme:str, email_conf, image_path=None):
    config = configparser.ConfigParser()
    config.read(email_conf)
    print(config.sections())
    # SMTP服务器,这里使用163邮箱
    mail_host = "smtp.163.com"
    # 发件人邮箱
    mail_sender = config["Mail"]["mail_sender"]
    # 邮箱授权码,注意这里不是邮箱密码,如何获取邮箱授权码,请看本文最后教程
    mail_license = config["Mail"]["mail_license"]
    # 收件人邮箱，可以为多个收件人
    mail_receivers = config["Mail"]["mail_receivers"]

    # 构建邮件对象
    mm = MIMEMultipart('related')

    # 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
    mm["From"] = config["Mail"]["mail_sender_withName"]
    # 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
    mm["To"] = config["Mail"]["mail_receivers_withName"]
    # 设置邮件主题
    mm["Subject"] = Header(theme, 'utf-8')

    # 邮件正文内容
    # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
    message_text = MIMEText(txt, "plain", "utf-8")
    # 向MIMEMultipart对象中添加文本对象
    mm.attach(message_text)

    if image_path is not None:
        with open(image_path, 'rb') as fp:
            picture = MIMEImage(fp.read())
            # 与txt文件设置相似
            picture['Content-Type'] = 'application/octet-stream'
            picture['Content-Disposition'] = 'attachment;filename="model_res.png"'
        mm.attach(picture)

    # 创建SMTP对象
    stp = smtplib.SMTP()
    # 设置发件人邮箱的域名和端口，端口地址为25
    stp.connect(mail_host, 25)
    # set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
    stp.set_debuglevel(1)
    # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
    stp.login(mail_sender, mail_license)
    # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
    stp.sendmail(mail_sender, mail_receivers, mm.as_string())
    print("邮件发送成功")
    # 关闭SMTP对象
    stp.quit()


# send_email(",,,",",,,", EMAIL_CONF)
