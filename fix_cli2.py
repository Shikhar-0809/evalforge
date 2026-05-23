content = open("evalforge/cli.py").read()
fixed = content.replace(
    '_ALLOWED_PROVIDERS = frozenset({"anthropic", "openai", "gemini"})',
    '_ALLOWED_PROVIDERS = frozenset({"anthropic", "openai", "gemini", "ollama"})'
)
open("evalforge/cli.py", "w").write(fixed)
print("Done")