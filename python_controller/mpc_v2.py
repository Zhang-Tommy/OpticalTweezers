def should_cut_wire(red, blue, star, led, serial_last_digit_even, parallel_port, batteries):
    # Decision tree logic based on conditions
    if red and blue and star and led:
        return 'D'  # Do not cut
    elif red and star and led:
        return 'B' if batteries >= 2 else 'D'
    elif red and blue:
        return 'P' if parallel_port else 'D'
    elif red and star:
        return 'C'
    elif red and led:
        return 'S' if serial_last_digit_even else 'D'
    elif blue and star:
        return 'B' if batteries >= 2 else 'D'
    elif blue and led:
        return 'P' if parallel_port else 'D'
    elif star:
        return 'S' if serial_last_digit_even else 'D'
    elif red:
        return 'C'
    elif blue:
        return 'P' if parallel_port else 'D'
    elif led:
        return 'D'
    else:
        return 'C'

# Example usage:
red = True  # Wire has red coloring
blue = False  # Wire has no blue coloring
star = True  # Wire has the ★ symbol
led = True  # LED is on
serial_last_digit_even = True  # Serial number's last digit is even
parallel_port = True  # Bomb has a parallel port
batteries = 3  # Bomb has three batteries

result = should_cut_wire(red, blue, star, led, serial_last_digit_even, parallel_port, batteries)
print(f"Decision: {result}")

should_cut_wire(red, blue, star, led, serial_last_digit_even, parallel_port, batteries)