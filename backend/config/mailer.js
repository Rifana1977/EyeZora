const nodemailer = require("nodemailer");

/**
 * EyeZora Centralized Nodemailer Transporter Configuration
 * 
 * Technical Investigation:
 * When Node.js tries to connect to Gmail's SMTP servers under local network environments,
 * it can encounter the "self-signed certificate in certificate chain" error.
 * This occurs because local security programs (Antivirus HTTPS scans like Bitdefender/Kaspersky)
 * or corporate proxy servers intercept encrypted TLS handshakes, replacing Gmail's certs
 * with local, self-signed root CA certificates.
 * 
 * Because Node.js doesn't query the native Windows OS Root Certificate Store automatically,
 * it rejects the certificate chain with this warning.
 * 
 * Safety & Best Practices Solution:
 * To handle this locally without compromising production servers, we conditionally set
 * `rejectUnauthorized` to false ONLY when NODE_ENV is not "production". In production,
 * TLS validation remains strictly enabled (rejectUnauthorized: true) for optimal security.
 */
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.EMAIL_FROM,
    pass: process.env.EMAIL_PASS,
  },
  tls: {
    // In local development, bypass self-signed cert chain verification.
    // In production, reject self-signed or unauthorized certificates.
    rejectUnauthorized: process.env.NODE_ENV === "production",
  },
});

/**
 * Sends an email notification to the student when their exam result is published.
 * Includes student details, marks obtained, percentage, and publication timestamp.
 */
transporter.sendResultPublishedEmail = async (student, result) => {
  const percentage = result.percentage;
  const isPass = percentage >= 50;
  const statusLabel = isPass ? "Pass" : "Fail";
  const statusColor = isPass ? "#10b981" : "#ef4444";
  const pubDate = new Date(result.publishedAt).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const mailOptions = {
    from: `"EyeZora Exam System" <${process.env.EMAIL_FROM}>`,
    to: student.email,
    subject: `EyeZora — Results Published: ${result.examTitle}`,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; background: #f8f7ff; border-radius: 12px;">
        <div style="text-align: center; margin-bottom: 24px;">
          <h1 style="color: #7c3aed; font-size: 26px; letter-spacing: 2px; margin: 0;">👁 EYEZORA</h1>
          <p style="color: #64748b; font-size: 13px; margin: 4px 0 0;">AI-Powered Examination System</p>
        </div>
        <div style="background: #fff; padding: 28px; border-radius: 10px; border: 1px solid rgba(124, 58, 237, 0.15); box-shadow: 0 4px 12px rgba(124, 58, 237, 0.03);">
          <h2 style="color: #1e1b4b; font-size: 18px; margin-top: 0; margin-bottom: 16px; border-bottom: 1px solid #f3f0ff; padding-bottom: 10px;">
            Your Examination Results are Published
          </h2>
          <p style="color: #374151; font-size: 14px; line-height: 1.7; margin: 0 0 16px;">
            Hi <strong>${student.name}</strong>,
          </p>
          <p style="color: #374151; font-size: 14px; line-height: 1.7; margin: 0 0 20px;">
            The results for your recent examination have been evaluated and published by the administrator. Here is a summary of your performance:
          </p>
          <div style="background: #f3f0ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid rgba(124, 58, 237, 0.2);">
            <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
              <tr>
                <td style="color: #64748b; padding: 6px 0; font-weight: 500;">Exam Name:</td>
                <td style="font-weight: 700; color: #1e1b4b; text-align: right; padding: 6px 0;">${result.examTitle}</td>
              </tr>
              <tr>
                <td style="color: #64748b; padding: 6px 0; font-weight: 500;">Marks Obtained:</td>
                <td style="font-weight: 700; color: #1e1b4b; text-align: right; padding: 6px 0;">${result.score} / ${result.totalMarks}</td>
              </tr>
              <tr>
                <td style="color: #64748b; padding: 6px 0; font-weight: 500;">Percentage:</td>
                <td style="font-weight: 700; color: #7c3aed; text-align: right; padding: 6px 0;">${percentage}%</td>
              </tr>
              <tr>
                <td style="color: #64748b; padding: 6px 0; font-weight: 500;">Status:</td>
                <td style="font-weight: 700; color: ${statusColor}; text-align: right; padding: 6px 0;">${statusLabel}</td>
              </tr>
              <tr>
                <td style="color: #64748b; padding: 6px 0; font-weight: 500;">Publication Date:</td>
                <td style="font-weight: 600; color: #1e1b4b; text-align: right; padding: 6px 0;">${pubDate}</td>
              </tr>
            </table>
          </div>
          <div style="text-align: center; margin-top: 24px;">
            <a href="${process.env.FRONTEND_URL || "http://localhost:3000"}/student/login"
               style="background: linear-gradient(135deg, #7c3aed, #5b21b6); color: #fff; padding: 12px 28px;
                      border-radius: 10px; text-decoration: none; font-weight: 700; font-size: 14px; display: inline-block;
                      box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35);">
              View Full Report
            </a>
          </div>
        </div>
        <div style="text-align: center; margin-top: 16px; color: #9ca3af; font-size: 11px;">
          This is an automated notification from EyeZora Examination System. Please do not reply directly to this email.
        </div>
      </div>
    `
  };

  return transporter.sendMail(mailOptions);
};

module.exports = transporter;
