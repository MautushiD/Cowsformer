import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def report(self) -> float:
        time_elapsed = time.time() - self.start
        print(
            "Time elapsed: {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        return time_elapsed


class BatchCounter:
    def __init__(self, num_batches: int):
        self.step = 1  # from 1 to num_batches
        self.num_batches = num_batches

    def report(self, loss: float) -> None:
        print(
            "\r", "Batch %d/%d: Loss %.3f" % (self.step, self.num_batches, loss), end=""
        )
        self.step += 1
