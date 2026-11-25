"""A shelve whose values are pickled and zlib-compressed.

NOTE: the thesis code imported `CompressShelve` from a separate `compress`
module that was not part of the archived code. This is a stand-in that
gives the same interface (dict-like, context manager, `flag="r"` for
read-only). Shelve files written by the original module may not be
readable with it.
"""
import pickle
import shelve
import zlib


class CompressShelve(shelve.Shelf):
    """Dict-like persistent store: str keys -> any picklable value, compressed on disk."""

    def __init__(self, filename, flag="c", protocol=pickle.HIGHEST_PROTOCOL, level=6):
        import dbm
        super().__init__(dbm.open(filename, flag), protocol=protocol, writeback=False)
        self._level = level

    def __getitem__(self, key):
        return pickle.loads(zlib.decompress(self.dict[key.encode(self.keyencoding)]))

    def __setitem__(self, key, value):
        data = zlib.compress(pickle.dumps(value, protocol=self._protocol), self._level)
        self.dict[key.encode(self.keyencoding)] = data

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
