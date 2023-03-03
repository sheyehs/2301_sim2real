from inspect import currentframe, getouterframes  # debugging
import time


class line_printer:
    def __init__(self) -> None:
        self.prev_time = time.time()

    def print_line(self):
        frame = currentframe()
        try:
            frameinfos = getouterframes(frame)
            frameinfo = frameinfos[1]
            curr_time = time.time()
            delta_time = curr_time - self.prev_time
            self.prev_time = curr_time
            print(f'{delta_time}\t{frameinfo.filename}\t{frameinfo.lineno}')
        finally:
            del frame, frameinfos
            
        return delta_time
