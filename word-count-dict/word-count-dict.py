from collections import Counter
def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    all_chr = []
    for i in sentences:
        all_chr += i
    return Counter(all_chr)
