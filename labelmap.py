# src/labelmap.py
# Define final emotion buckets and mapping rules (from GoEmotions labels -> 7 classes)

EMOTIONS = ["joy", "anger", "sadness", "fear", "love", "surprise", "neutral"]

# Groups of GoEmotions-style labels that map to each bucket (use lowercase)
JOY = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "optimism", "relief", "pride", "realization", "relief"
}
ANGER = {"anger", "annoyance", "disapproval", "disgust", "rage", "frustration"}
SADNESS = {"sadness", "disappointment", "embarrassment", "grief", "remorse"}
FEAR = {"fear", "nervousness", "anxiety", "stress", "dread", "panic", "worry", "confusion"}
LOVE = {"love", "affection", "fondness", "longing", "desire", "caring"}
SURPRISE = {"surprise", "curiosity", "amazement", "awe", "realization"}
NEUTRAL = {"neutral", "no_emotion", "other", "uncertainty"}

# Composite mapping function
def map_label_name_to_bucket(label_name: str) -> str:
    """
    Map a fine-grained label name to one of the EMOTIONS buckets.
    label_name: lowercase string
    """
    ln = label_name.lower().strip()
    if ln in JOY:
        return "joy"
    if ln in ANGER:
        return "anger"
    if ln in SADNESS:
        return "sadness"
    if ln in FEAR:
        return "fear"
    if ln in LOVE:
        return "love"
    if ln in SURPRISE:
        return "surprise"
    # default fallback
    return "neutral"

def bucket_name_to_index(name: str) -> int:
    return EMOTIONS.index(name)