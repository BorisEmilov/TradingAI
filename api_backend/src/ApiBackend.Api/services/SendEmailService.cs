using System;
using MailKit.Net.Smtp;
using MailKit.Security;
using MimeKit;
using Microsoft.Extensions.Options;


namespace ApiBackend.Api.Services
{
    public class SendEmailService
    {
        private readonly IConfiguration _config;

        public SendEmailService(IConfiguration config)
        {
            _config = config;
        }

        public async Task SendEmailAsync(string toEmail, string subject, string body)
        {
            var message = new MimeMessage();
            var settings = _config.GetSection("EmailSettings");
            message.From.Add(new MailboxAddress(settings["SenderName"], settings["SenderEmail"]));
            message.To.Add(new MailboxAddress("", toEmail));
            message.Subject = subject;
            message.Body = new TextPart("html") { Text = body };

            using var client = new SmtpClient();
            await client.ConnectAsync(settings["SmtpHost"], int.Parse(settings["SmtpPort"]!), SecureSocketOptions.StartTls);
            await client.AuthenticateAsync(settings["Username"], settings["Password"]);
            await client.SendAsync(message);
            await client.DisconnectAsync(true);
        }

        private static string CodeBody(string title, string intro, string code) => $@"
           <html>
           <body style='font-family: Arial, sans-serif;'>
               <h2>{title}</h2>
               <p>{intro}</p>
               <p style='font-size:32px; font-weight:bold; letter-spacing:8px;'>{code}</p>
               <p>The code expires in 10 minutes. If you did not request it, ignore this email.</p>
           </body>
           </html>";

        public async Task SendVerificationEmailAsync(string toEmail, string code)
        {
            var body = CodeBody("¡Welcome!", "Enter this code in the app to verify your email:", code);
            await SendEmailAsync(toEmail, "Email verification code", body);
        }

        public async Task SendPasswordChangeCodeAsync(string toEmail, string code)
        {
            var body = CodeBody("Password change", "Enter this code in the app to change your password:", code);
            await SendEmailAsync(toEmail, "Password change code", body);
        }

    };

}