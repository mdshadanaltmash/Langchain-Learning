import time


def retry(retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        raise e

        return wrapper

    return decorator


@retry(retries=3, delay=5)
def add_numbers(a, b):
    return a + b


# Example usage:
result = add_numbers(5, 3)
print(result)
result = add_numbers(5, '3')
print(result)
