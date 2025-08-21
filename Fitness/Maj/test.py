from openai import OpenAI, OpenAIError

# Hardcoded API key (⚠️ don’t commit this to GitHub!)
api_key = "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"

print("🔑 Using API key prefix:", api_key[:10] + "...")

client = OpenAI(api_key=api_key)

# 1. List models available to this key
try:
    models = client.models.list()
    print("\n✅ Models visible to this API key:")
    for m in models.data[:20]:  # show first 20
        print("-", m.id)
except OpenAIError as e:
    print("\n❌ Could not list models:", e)

# 2. Simple test query against gpt-4o
try:
    response = client.responses.create(   # modern API endpoint
        model="gpt-4o",                   # use "gpt-4o-mini" if 4o not visible
        input=[{"role": "user", "content": "Say hello"}]
    )
    print("\nTest GPT-4o response:", response.output_text)
except Exception as e:
    print("\n❌ GPT-4o test failed:", e)
