import pandas as pd

def standardize_format_name(format_name):
    """Map raw format names to standardized display names"""
    # Handle None or empty values
    if not format_name or pd.isna(format_name):
        return "Unknown"
    
    # Clean the format name (strip whitespace and normalize case for comparison)
    clean_format = str(format_name).strip()
    
    # If empty after cleaning, return Unknown
    if not clean_format:
        return "Unknown"
    
    format_mapping = {
        # T20 International variations
        'Int20 Tournament': 'T20I',
        'T20I': 'T20I',
        'International T20': 'T20I',
        'T20 International': 'T20I',
        
        # ODI variations
        'ODI': 'ODI',
        'One Day International': 'ODI',
        '50 Over Match': 'ODI',
        
        # Test variations
        'Test Match': 'Test',
        'Test': 'Test',
        'Test Championship Final': 'Test',
        'International Test': 'Test',
        
        # First Class variations
        'FC League': 'First Class',
        'First Class': 'First Class',
        'FC': 'First Class',
        'Four Day': 'First Class',
        '4 Day': 'First Class',
        
        # List A / One Day variations
        '1 Day Cup': 'List A',
        'List A': 'List A',
        'One Day Cup': 'List A',
        '50 Over Cup': 'List A',
        
        # T20 Domestic variations
        'T20 League': 'T20',
        'T20': 'T20',
        'Twenty20': 'T20',
        'T20 Cup': 'T20',
        'T20 Blast': 'T20',
        
        # Youth/Under variations
        'Asia Trophy 20': 'T20U',
        'U19 ODI': 'U19 ODI',
        'U19 Test': 'U19 Test',
        'U19 T20': 'T20U',
        'Youth ODI': 'U19 ODI',
        'Youth Test': 'U19 Test',
        'Youth T20': 'T20U'
    }
    
    # First try exact match
    if clean_format in format_mapping:
        return format_mapping[clean_format]
    
    # Try case-insensitive matching for common variations
    for key, value in format_mapping.items():
        if clean_format.lower() == key.lower():
            return value
    
    # Try partial matching for common patterns
    clean_lower = clean_format.lower()
    
    # T20 patterns
    if 't20' in clean_lower or 'twenty' in clean_lower:
        if 'international' in clean_lower or 'int' in clean_lower:
            return 'T20I'
        elif any(word in clean_lower for word in ['youth', 'u19', 'under']):
            return 'T20U'
        else:
            return 'T20'
    
    # ODI patterns
    if any(word in clean_lower for word in ['odi', 'one day', '50 over']):
        if any(word in clean_lower for word in ['youth', 'u19', 'under']):
            return 'U19 ODI'
        else:
            return 'ODI'
    
    # Test patterns
    if 'test' in clean_lower:
        if any(word in clean_lower for word in ['youth', 'u19', 'under']):
            return 'U19 Test'
        else:
            return 'Test'
    
    # First Class patterns
    if any(word in clean_lower for word in ['first class', 'fc', 'four day', '4 day']):
        return 'First Class'
    
    # List A patterns
    if any(word in clean_lower for word in ['list a', 'cup', '1 day']):
        return 'List A'
    
    # If no match found, return the original (cleaned) format name
    return clean_format

# Test the function
test_formats = ['Int20 Tournament', 'Asia Trophy 20', 'Test Championship Final', 'T20I', 'ODI', '', None, '  ', 'Unknown Format']

print("Testing format standardization function:")
print("=" * 50)
for fmt in test_formats:
    result = standardize_format_name(fmt)
    print(f"'{fmt}' -> '{result}'")
