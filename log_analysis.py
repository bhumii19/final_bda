from mrjob.job import MRJob
from mrjob.step import MRStep
from datetime import datetime
import json

class LogAnalysis(MRJob):

    def mapper(self, _, line):
        parts = line.strip().split()
        if len(parts) == 3:
            user, timestamp, action = parts
            yield user, json.dumps((timestamp, action))

    def reducer_total_time(self, user, values):
        events = []
        for v in values:
            timestamp, action = json.loads(v)
            events.append((timestamp, action))

        total = 0
        login_time = None

        for timestamp, action in sorted(events):
            try:
                t = datetime.fromisoformat(timestamp)
            except ValueError:
                continue

            if action == "login":
                login_time = t
            elif action == "logout" and login_time:
                total += (t - login_time).total_seconds()
                login_time = None

        # Emit user and their total login hours
        yield "user_hours", (user, round(total / 3600, 2))

    def reducer_find_max(self, key, user_time_pairs):
        user_hours = []
        max_time = 0
        top_users = []

        for user, hours in user_time_pairs:
            # Keep all users for printing
            user_hours.append((user, hours))

            # Find maximum
            if hours > max_time:
                max_time = hours
                top_users = [user]
            elif hours == max_time:
                top_users.append(user)

        # First print all users and their hours
        yield "All users and their total login hours", user_hours

        # Then print max user(s)
        yield "User(s) with maximum login time", top_users
        yield "Maximum login hours", max_time

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer_total_time),
            MRStep(reducer=self.reducer_find_max)
        ]

if __name__ == "__main__":
    LogAnalysis.run()
