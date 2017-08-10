class BracketError(RuntimeError):
	def __init__(self):
		pass

class UnopenedBracketError(BracketError):
	def __init__(self):
		pass

class UnclosedBracketError(BracketError):
	def __init__(self):
		pass
