import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def send_request(image_bytes):
    _ = requests.post(
        'http://localhost:8000/recognize/file',
        headers={'accept': 'application/json'},
        files={'file': ('test_image.png', image_bytes, 'image/png')}
    )


def main():
    # Arrange
    file_name = 'test_image.png'
    image_bytes = (Path(__file__).parent / file_name).read_bytes()
    num_requests = 1000

    # Act
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        for i in range(num_requests):
            executor.submit(send_request, image_bytes)
    end_time = time.time()

    # Assert
    total_time = end_time - start_time
    avg_time_per_request = (total_time / num_requests) * 1000
    print(f'\nTotal time taken for {num_requests} requests: {total_time:.2f} seconds')
    print(f'Average time per request: {avg_time_per_request:.2f} milliseconds')


if __name__ == '__main__':
    main()
