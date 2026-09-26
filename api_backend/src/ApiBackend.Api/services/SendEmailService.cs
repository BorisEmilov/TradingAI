using System;
using MailKit.Net.Smtp;
using MailKit.Security;
using MimeKit;
using Microsoft.Extensions.Options;


namespace ApiBackend.Api.Services
{
    public class SendEmailService
    {
        private const string SmtpHost = "smtp.gmail.com";
        private const int SmtpPort = 587;
        private const string SenderEmail = "companyEmail";
        private const string SenderName = "CompanyName";
        private string Password = "App_password";

        public async Task SendEmailAsync(string toEmail, string subject, string body)
        {
            var message = new MimeMessage();
            message.From.Add(new MailboxAddress(SenderName, SenderEmail));
            message.To.Add(new MailboxAddress("", toEmail));
            message.Subject = subject;
            message.Body = new TextPart("html") { Text = body };

            using var client = new SmtpClient();
            await client.ConnectAsync(SmtpHost, SmtpPort, SecureSocketOptions.StartTls);
            await client.AuthenticateAsync(SenderEmail, Password);
            await client.SendAsync(message);
            await client.DisconnectAsync(true);
        }

        public async Task SendVerificationEmailAsync(string toEmail, string verifyUrl)
        {
            var verificationLink = verifyUrl;


            var body = $@"
           <html>
           <body style='font-family: Arial, sans-serif;'>
               <h2>¡Wellcome!</h2>
               <p>Please verify your email by clicking the following link:</p>
               <a href='{verificationLink}' style='background:#4CAF50; color:white; padding:12px 20px; border-radius:4px; text-decoration:none;'>
                   Verify yout email
               </a>
               <p>The link expires in 10 minutes</p>
           </body>
           </html>";


            await SendEmailAsync(toEmail, "no-replay Email Verification", body);
        }


        public async Task SendPasswordChangeLink(string toEmail, string verifyUrl)
        {
            var verificationLink = verifyUrl;


            var body = $@"
           <html>
           <body style='font-family: Arial, sans-serif;'>
               <h2>¡Wellcome!</h2>
               <p>Click the link to cjange your password:</p>
               <a href='{verificationLink}' style='background:#4CAF50; color:white; padding:12px 20px; border-radius:4px; text-decoration:none;'>
                   Change your password
               </a>
               <p>The link expires in 10 minutes</p>
           </body>
           </html>";


            await SendEmailAsync(toEmail, "no-replay Password Change", body);
        }

    };

}