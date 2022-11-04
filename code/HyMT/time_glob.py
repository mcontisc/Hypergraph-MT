import time
import types
from warnings import warn


class Printer:
    def __init__(self):
        self.verboseness = False
        self.verbosity = 0
        self.verbose_key = ""

    def __call__(self, *args, **kwargs):
        if self.verboseness:
            key = (
                next(self.verbose_key)
                if isinstance(self.verbose_key, types.GeneratorType)
                else self.verbose_key
            )
            if self.verbosity == 0:
                print()
            print(self.verbosity * "   " + key + ":", *args, **kwargs)


class Timer:
    def __init__(
        self,
        name="timer",
        color="blue",
        level=0,
        verbose=False,
        print_details=False,
        print_sub=True,
    ):
        self.name = name
        self.color = color
        self.level = level
        self.zero = time.time()
        self.last = self.zero
        self.sub = None
        self.rec = dict()
        self.verbose = verbose
        self.print_details = print_details
        self.print_sub = print_sub

    def cprint(self, *args, **kwargs):
        if self.verbose:
            # print(colored(*args, self.color), **kwargs) if len(args) > 0 else print()
            print((*args, self.color), **kwargs) if len(args) > 0 else print()

    def print_time(self, string, t, **kwargs):
        self.cprint(string + ": %.3f seconds" % t, **kwargs)

    def __call__(self, string=None, t=None):
        if string is not None:
            self.print_time(
                self.level * "  " + string,
                t=(time.time() - self.last) if t is None else t,
                end="\n\n" if self.level == 0 else "\n",
            )
        self.last = time.time()

    def total(self):
        return time.time() - self.zero

    def start(self, name):
        if self.sub is not None:
            self.sub.start(name)
        else:
            self.sub = Timer(
                name=name,
                color=self.color,
                level=self.level + 1,
                verbose=self.verbose,
                print_details=self.print_details,
            )

    def print_rec(self, d=None, tab=4, current_t=0):
        if d is None:
            d = self.rec
        if len(d) > 0:
            max_len = max(len(k) for k in d.keys())
            d = dict(zip((tab * " " + k.ljust(max_len) for k in d.keys()), d.values()))
            for k, v in d.items():
                if isinstance(v, dict):
                    if "t" in v:
                        t = v.pop("t")
                        self.print_time(k, t=t, end="")
                        self.cprint(" (%.0f%%" % (100 * t / current_t), end="")
                    if "n" in v:
                        n = v.pop("n")
                        if self.print_details:
                            self.cprint(
                                f", calls = {n}" + ", avg = %.6f" % (t / n), end=""
                            )
                    self.cprint(")")
                    if "rec" in v:
                        self.print_rec(v["rec"], tab=tab + max_len + 2, current_t=t)
                else:
                    self.print_time(k, t=v)

    def stop(self, name):
        if name != self.sub.name:
            if self.sub is not None:
                self.sub.stop(name)
            else:
                warn("trying to stop a timer that never started")
        else:
            if self.level == 0:
                t = self.sub.total()
                self.print_time(self.sub.name, t=t)
                if self.print_sub:
                    self.sub.print_rec(current_t=t)
                self.cprint()
            else:
                self.rec[self.sub.name] = {"t": self.sub.total(), "rec": self.sub.rec}
            self.sub = None
            self.last = time.time()

    def cum(self, name):
        if self.sub is not None:
            self.sub.cum(name)
        else:
            if name not in self.rec.keys():
                self.rec[name] = {"t": 0, "n": 0}
            self.rec[name]["t"] += time.time() - self.last
            self.rec[name]["n"] += 1
            self.last = time.time()

    def start_cum(self, name):
        if self.sub is not None:
            self.sub.start_cum(name)
        else:
            self.sub = Timer(
                name=name,
                color=self.color,
                level=self.level + 1,
                verbose=self.verbose,
                print_details=self.print_details,
            )
            if name not in self.rec.keys():
                self.rec[name] = {"t": 0, "n": 0}
            else:
                self.sub.rec = self.rec[name].pop("rec")

    def stop_cum(self, name):
        if name != self.sub.name:
            if self.sub is not None:
                self.sub.stop_cum(name)
            else:
                warn("trying to stop a timer that never started")
        else:
            self.rec[self.sub.name]["t"] += self.sub.total()
            self.rec[self.sub.name]["n"] += 1
            self.rec[self.sub.name]["rec"] = self.sub.rec
            self.sub = None
            self.last = time.time()


timer = Timer()
printer = Printer()
prng = None
parallelize = False


def timeit(name):
    def dec(func):
        def wrap(*a, **k):
            timer.start(name)
            res = func(*a, **k)
            timer.stop(name)
            return res

        return wrap

    return dec


def timeit_cum(name):
    def dec(func):
        def wrap(*a, **k):
            timer.start_cum(name)
            res = func(*a, **k)
            timer.stop_cum(name)
            return res

        return wrap

    return dec
