class ProtonObject:
    def __init__(self, cname, attrs):
        self.cname = cname
        self.attrs = attrs
    def __getattr__(self, attrname):
        return self.attrs[attrname]
    def __setattr__(self, attrname, value):
        self.attrs[attrname] = value
    def __getitem__(self, index):
        return self.attrs[index]
    def __setitem__(self, index, value):
        self.attrs[index] = value
    def __repr__(self):
        return self.cname + " " + str(self.attrs)
