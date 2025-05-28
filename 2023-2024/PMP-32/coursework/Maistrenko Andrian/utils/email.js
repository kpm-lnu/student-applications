const nodemailer = require("nodemailer");
const sgMail = require("@sendgrid/mail");
const nodemailerSendgrid = require("nodemailer-sendgrid");
const Transport = require("nodemailer-brevo-transport");
const pug = require("pug");
const { convert } = require("html-to-text");

sgMail.setApiKey(process.env.SENDGRID_USERNAME);

module.exports = class Email {
  constructor(user, url) {
    this.to = user.email;
    this.firstName = user.name.split(" ")[0];
    this.url = url;
    this.from = `Andrian Maistrenko <${process.env.EMAIL_FROM}>`;
  }

  newTransport() {
    if (process.env.NODE_ENV === "production") {
      //sendgrid
      // return nodemailer.createTransport({
      //     service: 'SendGrid',
      //     auth:{
      //         user: process.env.SENDGRID_USERNAME,
      //         pass: process.env.SENDGRID_PASSWORD
      //     }
      // });
      // return {
      //     send: (msg) => sgMail.send(msg)
      // };
      // return nodemailer.createTransport(
      //     nodemailerSendgrid({
      //       apiKey: process.env.SENDGRID_USERNAME
      //     })
      // )
      // console.log(
      //   "BREVO_API_KEY:",
      //   process.env.BREVO_API_KEY ? "Key is set" : "Key is missing",
      // );
      return nodemailer.createTransport(
        new Transport({ apiKey: process.env.BREVO_API_KEY }),
      );
      // return nodemailer.createTransport({
      //     host: 'smtp-relay.brevo.com',
      //     port:587,
      //     secure:false,
      //     auth:{
      //         user:
      //         pass:
      //     }
      // });
    }
    return nodemailer.createTransport({
      host: process.env.EMAIL_HOST,
      port: process.env.EMAIL_PORT,
      auth: {
        user: process.env.EMAIL_USERNAME,
        pass: process.env.EMAIL_PASSWORD,
      },
      // activate in gmail "less secure app" option for gmail
    });
  }

  //send the email
  async send(template, subject) {
    const html = pug.renderFile(`${__dirname}/../views/email/${template}.pug`, {
      firstName: this.firstName,
      url: this.url,
      subject,
    });
    //render pug temp

    //define options
    const mailOptions = {
      from: this.from,
      to: this.to,
      subject,
      html,
      text: convert(html),
    };
    //create transport and send
    try {
      await this.newTransport().sendMail(mailOptions);
    } catch (error) {
      console.error("Error sending email:", error);
      if (error.response) {
        console.error("Error response:", error.response.body);
      }
      throw new Error("There was an error sending the email. Try again later.");
    }
  }

  async sendWelcome() {
    await this.send("welcome", "Welcome to the Computer Shop");
  }

  async sendPasswordReset() {
    await this.send(
      "passwordReset",
      "Your password reset token (valid only for 10 minutes)",
    );
  }
};
