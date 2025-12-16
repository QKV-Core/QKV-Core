# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities via email to:

**Email:** huseyinkama79@outlook.com

### What to Include

When reporting a security vulnerability, please include:

1. **Description**: A clear description of the vulnerability
2. **Impact**: The potential impact of the vulnerability
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Proof of Concept**: If possible, include a proof of concept or exploit code
5. **Suggested Fix**: If you have ideas for how to fix the issue, please share them

### Response Timeline

- **Initial Response**: We will acknowledge receipt of your report within 48 hours
- **Status Update**: We will provide a status update within 7 days
- **Resolution**: We will work to resolve critical vulnerabilities as quickly as possible

### Disclosure Policy

- We will credit you for discovering the vulnerability (unless you prefer to remain anonymous)
- We will work with you to coordinate public disclosure after a fix is available
- We will not disclose your identity without your explicit permission

## Security Best Practices

When using QKV Core, please follow these security best practices:

1. **Keep Dependencies Updated**: Regularly update all dependencies to the latest secure versions
2. **Validate Input**: Always validate and sanitize user inputs, especially when loading models or processing data
3. **Secure Model Storage**: Store model files in secure locations with appropriate access controls
4. **Environment Variables**: Never commit sensitive credentials or API keys to version control
5. **Network Security**: When downloading models from external sources, verify checksums and use HTTPS

## Known Security Considerations

- **Model Loading**: Loading untrusted model files can execute arbitrary code. Only load models from trusted sources.
- **Web UI**: The Gradio web interface should not be exposed to untrusted networks without proper authentication.
- **Database Connections**: Use secure connection strings and never expose database credentials.

## Security Updates

Security updates will be announced through:
- GitHub Security Advisories
- Release notes
- Email notifications (for critical vulnerabilities)

Thank you for helping keep QKV Core secure!

