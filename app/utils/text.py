def normalize_text(text: str) -> str:
    """
    Normalize text by converting Turkish characters to their ASCII equivalents
    """
    turkish_chars = {
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    }
    for tr, eng in turkish_chars.items():
        text = text.replace(tr, eng)
    return text
