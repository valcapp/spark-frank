
def make_lineparser(*idx_types):
    def lineparse(data):
        return tuple(typ(data[idx])
                    for idx, typ in idx_types)
    return lineparse

