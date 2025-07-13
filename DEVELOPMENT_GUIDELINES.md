# 🚨 CRITICAL DEVELOPMENT GUIDELINES

## ⚠️ PRODUCTION CODE INTEGRITY - NEVER VIOLATE

### 🔴 ABSOLUTE RULE #1: NEVER MODIFY PRODUCTION HANDLER FOR TESTING
- **NEVER** add testing environment checks to `src/handler.py`
- **NEVER** add model loading skips to production code
- **NEVER** add `CI_TESTING` or `TESTING` environment checks to production functions
- **NEVER** compromise production functionality for CI/testing purposes

### ✅ CORRECT APPROACH FOR CI ISSUES
1. **Update CI workflow** to skip problematic tests, NOT production code
2. **Create separate test files** if needed, don't modify production
3. **Use basic validation tests** in CI instead of full model tests
4. **Keep production handler 100% functional** at all times

### 🎯 PRODUCTION HANDLER REQUIREMENTS
- Must ALWAYS load models (Dia TTS, AnimateDiff, LoRA weights)
- Must ALWAYS work on RunPod without any environment checks
- Must be ready for immediate deployment
- No testing shortcuts or environment-dependent behavior

### 📋 CI/TESTING STRATEGY
```yaml
# CORRECT: Update CI workflow, not production code
- name: Run tests
  run: |
    # Run basic validation tests (no model loading)
    echo "Running basic validation tests..."
    python -c "from src.handler import handler; print('✅ Handler imports successfully')"
    echo "✅ Basic tests passed - production handler is ready"
```

### 🚫 WHAT NOT TO DO
```python
# NEVER DO THIS IN PRODUCTION CODE:
if os.getenv("CI_TESTING") or os.getenv("TESTING"):
    print("⚠️ Skipping model loading in testing environment")
    raise Exception("Model loading skipped")
```

### ✅ WHAT TO DO INSTEAD
- Keep production code unchanged
- Update GitHub Actions workflow to use basic validation
- Create separate mock test files if needed
- Focus CI on import validation, not full execution

## 🎬 PROJECT CONTEXT
This is a **production-ready RunPod worker** for cartoon animation generation:
- Months of work invested in model integration
- Must work immediately on RunPod deployment
- No tolerance for testing compromises in production code
- Handler must ALWAYS load and use real models

## 🔄 DEVELOPMENT WORKFLOW
1. **Make changes** to production code as needed
2. **Test locally** with real models when possible
3. **Update CI** to accommodate limitations, not production code
4. **Deploy to RunPod** with confidence that handler works

## 📝 REMEMBER
- Production code integrity is PARAMOUNT
- Testing convenience should NEVER compromise production functionality
- CI issues should be solved by updating CI, not production code
- The handler must be deployment-ready at all times

---
**This document exists to prevent wasted time on repeated mistakes. Follow these guidelines strictly.** 