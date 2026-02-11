# Security Policy

## Reporting a Security Vulnerability

**DO NOT** create a public issue for security vulnerabilities.

Instead, please email your findings to: `security@example.com` with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you on a fix.

## Security Considerations

### Input Validation

The system validates all user inputs:

- **File uploads**: PDF only, max 100MB
- **Queries**: Maximum 1000 characters
- **File paths**: Sanitization against directory traversal
- **JSON bodies**: Pydantic validation
- **API parameters**: Type and range checking

### Data Protection

- **No authentication**: Designed for internal use
- **No secrets in logs**: API keys/credentials excluded
- **HTTPS ready**: Supports TLS deployment
- **CORS configurable**: Can restrict to trusted origins

### Resource Protection

- **Process timeouts**: All LLM calls have 60-second timeout
- **Memory limits**: Configurable batch sizes
- **Disk limits**: Max index file size enforced
- **Concurrent operations**: Single index to prevent race conditions

## Deployment Security

### Production Checklist

- [ ] Enable HTTPS/TLS
- [ ] Add authentication (API key or OAuth)
- [ ] Configure CORS to trusted origins only
- [ ] Use environment variables for secrets
- [ ] Enable audit logging
- [ ] Configure firewall rules
- [ ] Run on non-root user
- [ ] Use read-only file mounts where possible
- [ ] Implement rate limiting
- [ ] Set up security monitoring

### Environment Variable Security

Never commit `.env` files. Use `.env.example` as template.

```bash
# .gitignore
.env
.env.local
.env*.local
```

### Docker Security

```dockerfile
# Use specific base image version (not latest)
FROM python:3.11-slim

# Run as non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

## Integration Security

### Ollama (LLM Provider)

- Local inference only - no requests to external servers
- Runs on localhost:11434 by default
- Should be internal network only in production

### Vector Store (FAISS)

- No network operations
- All data stored locally
- Consider encryption-at-rest for sensitive data

### PDF Processing

- Temporary files cleaned up after processing
- No personal data retained in indexes
- Consider data deletion policies

## Vulnerability Disclosure

When we receive a security report:

1. **Acknowledge** within 48 hours
2. **Investigate** the vulnerability
3. **Develop** a fix
4. **Test** the fix thoroughly
5. **Release** patch version with fix
6. **Notify** reporter of release
7. **Consider** CVE assignment if severe

## Known Limitations

### Current Version (1.0.0)

- No built-in authentication
- CORS allows all origins by default
- No rate limiting
- No request signing/verification

These should be addressed before production deployment.

## Security Best Practices

### For Users

1. **Run on internal networks** only (for now)
2. **Keep Ollama updated** to latest version
3. **Monitor disk space** for index files
4. **Validate extracted data** before using in critical systems
5. **Maintain backups** of vector indexes

### For Developers

1. **Never log sensitive data** (API keys, passwords)
2. **Sanitize all outputs** before returning to users
3. **Keep dependencies updated**: `pip list --outdated`
4. **Use type hints** to catch errors early
5. **Write security-focused tests**

## Third-Party Dependencies

Security considerations for major dependencies:

| Package | Purpose | Security Notes |
|---------|---------|-----------------|
| FastAPI | REST API | Well-maintained, security-first |
| FAISS | Vector search | No network access, data is local |
| sentence-transformers | Embeddings | Model downloads from HuggingFace |
| PyMuPDF | PDF parsing | Can be resource-intensive |
| Ollama | LLM inference | Local only, very safe |

## Compliance

This system handles technical data and PDFs:

- **GDPR**: Implement data deletion policies
- **PCI-DSS**: Not applicable (no payment data)
- **HIPAA**: Not applicable (no health data)
- **SOC 2**: Consider for cloud deployment

## Code Review

All pull requests undergo security review:

- Input validation checks
- Output sanitization
- Dependency updates
- Error handling for edge cases
- Resource exhaustion prevention

## Updates and Patches

Subscribe to releases for security updates:

```bash
# Watch releases
git clone [repo] && cd vehicle-spec-rag
git remote add upstream [repo]
git fetch upstream
```

Critical security updates will be marked as such in CHANGELOG.md.

## Questions?

For security questions (non-vulnerability): `security@example.com`

---

**Last Updated**: February 2025
**Status**: Security-reviewed v1.0
