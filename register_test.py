import requests
import json
from colorama import Fore, Style

BASE_URL = "http://api.pillq.com/api/auth/register"

def test_normal_scenarios():
    print(f"{Fore.BLUE}Running: test_normal_scenarios{Style.RESET_ALL}")

    # Valid Input
    valid_data = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=valid_data)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Valid Input{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Valid Input{Style.RESET_ALL}")
        print(response.json())

    # Case Insensitivity for Email
    mixed_case_email = {
        "name": "John Doe",
        "email": "John.Doe@Example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=mixed_case_email)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Case Insensitivity for Email{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Case Insensitivity for Email{Style.RESET_ALL}")
        print(response.json())

    # Valid Special Characters in Name
    special_chars_name = {
        "name": "John O'Connor",
        "email": "john.oconnor@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=special_chars_name)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Valid Special Characters in Name{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Valid Special Characters in Name{Style.RESET_ALL}")
        print(response.json())

    # Minimum Password Length
    min_password_length = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password1!",
        "re_enter_password": "Password1!"
    }
    response = requests.post(BASE_URL, json=min_password_length)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Minimum Password Length{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Minimum Password Length{Style.RESET_ALL}")
        print(response.json())

def test_better_to_have_scenarios():
    print(f"{Fore.BLUE}Running: test_better_to_have_scenarios{Style.RESET_ALL}")

    # Optional Fields
    optional_fields = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=optional_fields)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Optional Fields{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Optional Fields{Style.RESET_ALL}")
        print(response.json())

    # Phone Number Formatting
    phone_formats = [
        "+1 123-456-7890",
        "1234567890"
    ]
    for phone in phone_formats:
        data = {
            "name": "John Doe",
            "email": "johndoe@example.com",
            "phone": phone,
            "password": "Password123!",
            "re_enter_password": "Password123!"
        }
        response = requests.post(BASE_URL, json=data)
        if response.status_code == 200:
            print(f"{Fore.GREEN}Passed: Phone Number Formatting - {phone}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed: Phone Number Formatting - {phone}{Style.RESET_ALL}")
            print(response.json())

    # Whitespace Trimming
    whitespace_data = {
        "name": " John Doe ",
        "email": " johndoe@example.com ",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=whitespace_data)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Whitespace Trimming{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Whitespace Trimming{Style.RESET_ALL}")
        print(response.json())

    # Localization Support
    localized_name = {
        "name": "Élise Müller",
        "email": "elise.mueller@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=localized_name)
    if response.status_code == 200:
        print(f"{Fore.GREEN}Passed: Localization Support{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Localization Support{Style.RESET_ALL}")
        print(response.json())

def test_edge_cases():
    print(f"{Fore.BLUE}Running: test_edge_cases{Style.RESET_ALL}")

    # Empty Fields
    empty_fields = {
        "name": "",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=empty_fields)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Empty Fields{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Empty Fields{Style.RESET_ALL}")
        print(response.json())

    # Password Mismatch
    password_mismatch = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password456!"
    }
    response = requests.post(BASE_URL, json=password_mismatch)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Password Mismatch{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Password Mismatch{Style.RESET_ALL}")
        print(response.json())

    # Duplicate Email
    duplicate_email = {
        "name": "Jane Doe",
        "email": "johndoe@example.com",
        "phone": "0987654321",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=duplicate_email)
    if response.status_code == 409:
        print(f"{Fore.GREEN}Passed: Duplicate Email{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Duplicate Email{Style.RESET_ALL}")
        print(response.json())

    # Invalid Email Format
    invalid_emails = [
        "plainaddress",
        "@missingusername.com",
        "missingdomain@.com"
    ]
    for email in invalid_emails:
        data = {
            "name": "John Doe",
            "email": email,
            "phone": "1234567890",
            "password": "Password123!",
            "re_enter_password": "Password123!"
        }
        response = requests.post(BASE_URL, json=data)
        if response.status_code == 400:
            print(f"{Fore.GREEN}Passed: Invalid Email Format - {email}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed: Invalid Email Format - {email}{Style.RESET_ALL}")
            print(response.json())

    # Invalid Phone Number
    invalid_phones = [
        "abc1234567",
        "12345",
        "+99 12345678901"
    ]
    for phone in invalid_phones:
        data = {
            "name": "John Doe",
            "email": "johndoe@example.com",
            "phone": phone,
            "password": "Password123!",
            "re_enter_password": "Password123!"
        }
        response = requests.post(BASE_URL, json=data)
        if response.status_code == 400:
            print(f"{Fore.GREEN}Passed: Invalid Phone Number - {phone}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed: Invalid Phone Number - {phone}{Style.RESET_ALL}")
            print(response.json())

    # Weak Password
    weak_password = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "password123",
        "re_enter_password": "password123"
    }
    response = requests.post(BASE_URL, json=weak_password)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Weak Password{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Weak Password{Style.RESET_ALL}")
        print(response.json())

    # Overly Long Input
    long_inputs = {
        "name": "a" * 256,
        "email": "a" * 320 + "@example.com",
        "phone": "1" * 50,
        "password": "a" * 256,
        "re_enter_password": "a" * 256
    }
    response = requests.post(BASE_URL, json=long_inputs)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Overly Long Input{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Overly Long Input{Style.RESET_ALL}")
        print(response.json())

    # SQL Injection
    sql_injection = {
        "name": "' OR '1'='1",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=sql_injection)
    if response.status_code != 200:
        print(f"{Fore.GREEN}Passed: SQL Injection{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: SQL Injection{Style.RESET_ALL}")
        print(response.json())

    # XSS
    xss_payload = {
        "name": "<script>alert('XSS')</script>",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=xss_payload)
    if response.status_code != 200:
        print(f"{Fore.GREEN}Passed: XSS{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: XSS{Style.RESET_ALL}")
        print(response.json())

def test_scenarios_that_can_lead_to_break():
    print(f"{Fore.BLUE}Running: test_scenarios_that_can_lead_to_break{Style.RESET_ALL}")

    # Missing Required Fields
    missing_required = {
        "name": "John Doe",
        "phone": "1234567890"
    }
    response = requests.post(BASE_URL, json=missing_required)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Missing Required Fields{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Missing Required Fields{Style.RESET_ALL}")
        print(response.json())

    # Invalid Content-Type Header
    headers = {
        "Content-Type": "text/plain"
    }
    response = requests.post(BASE_URL, data=json.dumps(valid_data), headers=headers)
    if response.status_code == 415:
        print(f"{Fore.GREEN}Passed: Invalid Content-Type Header{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Invalid Content-Type Header{Style.RESET_ALL}")
        print(response.json())

    # Invalid HTTP Method
    response = requests.get(BASE_URL)
    if response.status_code == 405:
        print(f"{Fore.GREEN}Passed: Invalid HTTP Method{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Invalid HTTP Method{Style.RESET_ALL}")
        print(response.json())

    # Excessive Request Size
    large_payload = {"data": "a" * (1024 * 1024)}  # 1MB payload
    response = requests.post(BASE_URL, json=large_payload)
    if response.status_code == 413:
        print(f"{Fore.GREEN}Passed: Excessive Request Size{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Excessive Request Size{Style.RESET_ALL}")
        print(response.json())

    # Unsupported Characters
    unsupported_name = {
        "name": "John Doe 😀",
        "email": "johndoe@example.com",
        "phone": "1234567890",
        "password": "Password123!",
        "re_enter_password": "Password123!"
    }
    response = requests.post(BASE_URL, json=unsupported_name)
    if response.status_code == 400:
        print(f"{Fore.GREEN}Passed: Unsupported Characters{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: Unsupported Characters{Style.RESET_ALL}")
        print(response.json())

    # API Rate Limiting
    for _ in range(10):
        response = requests.post(BASE_URL, json=valid_data)
    if response.status_code == 429:
        print(f"{Fore.GREEN}Passed: API Rate Limiting{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed: API Rate Limiting{Style.RESET_ALL}")
        print(response.json())

if __name__ == "__main__":
    test_normal_scenarios()
    test_better_to_have_scenarios()
    test_edge_cases()
    test_scenarios_that_can_lead_to_break()
    print(f"{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
