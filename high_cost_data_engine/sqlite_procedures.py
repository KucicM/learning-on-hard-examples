from math import sqrt


class StandardDeviation:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_square = 0

    def step(self, value):
        if value is None:
            return

        self.count += 1
        self.sum += value
        self.sum_square += value ** 2

    def finalize(self):
        if self.count == 0:
            return 0
        mean = self.sum / self.count
        mean_squared = self.sum_square / self.count
        variance = mean_squared - mean ** 2
        return sqrt(variance)


def init_procedures(connection):
    connection.create_aggregate("STD", 1, StandardDeviation)
