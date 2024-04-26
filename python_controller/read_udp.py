import socket
import time


def read_from_port_to_file(duration_seconds=20):
    """Reads from port 61556 and writes the received data to 'shader_source.txt' for a specified duration"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 61557))

    try:
        start_time = time.time()
        previous_data = b''  # Initialize with an empty byte string

        while time.time() - start_time < duration_seconds:
            # Receive data from the port
            data, _ = server_socket.recvfrom(8192)  # Adjust buffer size as needed

            # Check if the received data is different from the previous data
            if data != previous_data:
                # Write the received data to 'shader_source.txt'
                with open('shader_source.txt', 'ab') as file:  # Use 'ab' to append binary data
                    file.write(data)
                    file.write(b"\n")  # Add a newline

                print("Data written to 'shader_source.txt'")
                previous_data = data  # Update the previous_data with the current data

    except Exception as e:
        print(f"Error: {e}")

    finally:
        server_socket.close()


# Call the function to read from the port and write to the file for 20 seconds
read_from_port_to_file(duration_seconds = 20)
