import re, ast, sys

from errors import *

class Token:
	def __init__(self, type, content, **kwargs):
		self.type = type
		self.content = content
		self.values = kwargs
	def type(self):
		return self.type
	def content(self):
		return self.content
	def values(self):
		return self.values
	def getValue(self, key):
		return self.values[key]
	def __str__(self):
		return '<Token type=%s content=%s>' % (str(self.type), str(self.content))
	def __repr__(self):
		return str(self)
	def __eq__(self, other):
		return isinstance(other, Token) and other.type == self.type and other.content == self.content

class LexerMatcher:
	def __init__(self, matcher, getter, skip = lambda *a: 0):
		self.matcher = matcher
		self.getter = getter
		self.skip = skip
	def match(self, code):
		return self.matcher(code)
	def get(self, code, match):
		return self.getter(code, match)
	def skip(self, code, match):
		return self.skip(code, match)

class RegexMatcher(LexerMatcher):
	def __init__(self, regex, group, tokentype, processor = lambda x: x, **kwargs):
		self.regex = regex
		self.group = group
		self.tokentype = tokentype
		self.processor = processor
		self.values = kwargs
	def match(self, code):
		return re.match(self.regex, code)
	def get(self, code, match):
		return (match.span()[1], Token(self.tokentype, self.processor(match.group(self.group)), **self.values))
	def skip(self, code, match):
		return match.span()[1] * (self.group == -1)

class ErrorMatcher(LexerMatcher):
	def __init__(self, matcher, errortype):
		self.matcher = matcher
		self.errortype = errortype
	def match(self, code):
		return self.matcher.match(code)
	def get(self, code, match):
		raise self.errortype()
	def skip(self, code, match):
		return self.matcher.skip(code, match)

def FT(key):
	print (key)
	return key

def oper_matcher(array, name, counter = []):
	return LexerMatcher(lambda code: [operator for operator in array if code.startswith(operator) and operator not in counter], lambda code, match: (max(map(len, match)), Token(name, max(match, key = len))))

class Lexer:
	def __init__(self, rules, code):
		self.rules = rules
		self.code = code
		self.index = 0
	def __iter__(self):
		return self
	def __next__(self):
		if self.index >= len(self.code): raise StopIteration
		for rule in self.rules:
			code = self.code[self.index:]
			match = rule.match(code)
			if match:
				skip = rule.skip(code, match)
				if skip:
					self.index += skip
					return self.__next__()
				else:
					token = rule.get(code, match)
					self.index += token[0]
					return token[1]
		raise RuntimeError('Unknown token at index %d: "...%s..."' % (self.index, self.code[self.index:][:10].replace('\n', '\\n')))

binary_RTL = [
	('**',),
	('=',)
]

binary_operators = [
	('.',),
	('**',),
	('*', '/', '//'),
	('+', '-'),
	('>>', '<<'),
	('>', '<', '<=', '>='),
	('**=', '*=', '/=', '//=', '+=', '-=', '>>=', '<<='),
	('==', '!=', ':=', '='),
	('in', 'not in',),
	('and',),
	('or',),
]

unifix_operators = ['!', '++', '--', '~']

class Pipe:
	def __init__(self, function):
		self.function = function
	def __ror__(self, other):
		return self.function(other)
	def __call__(self, *args, **kwargs):
		return Pipe(lambda x: self.function(x, *args, **kwargs))

def recurstr(array):
	if isinstance(array, (map,)):
		array = list(array)
	if isinstance(array, list):
		return str(list(map(recurstr, array)))
	return str(array)

def FT(key):
	print(recurstr(key))
	return key

matchers = [
	RegexMatcher(r'#.+', -1, 'comment'),
	RegexMatcher(r'/\*([^*]|\*[^/])*\*/', -1, 'comment'),
	RegexMatcher(r'(\d+\.\d*|\d*\.\d+)', 0, 'literal:expression', float),
	RegexMatcher(r'\d+', 0, 'literal:expression', int),
	RegexMatcher(r'"([^"\\]|\\.)*"', 0, 'literal:expression', lambda x: ast.literal_eval('""%s""' % x)),
	RegexMatcher(r"'([^'\\]|\\.)*'", 0, 'literal:expression', lambda x: ast.literal_eval("''%s''" % x)),
	ErrorMatcher(RegexMatcher(r'"([^"\\]|\\.)*', 0, ''), UnclosedStringError),
	ErrorMatcher(RegexMatcher(r"'([^'\\]|\\.)*", 0, ''), UnclosedStringError),
	RegexMatcher(r'(if|else|unless|while|for|try|except)', 0, 'keyword'),
	oper_matcher(unifix_operators, 'unifix_operator'),
	oper_matcher(sum(binary_operators, ()), 'binary_operator', sum(binary_RTL, ())),
	oper_matcher(sum(binary_RTL, ()), 'binary_RTL'),
	RegexMatcher(r'[A-Za-z_][A-Za-z_0-9]*', 0, 'identifier:expression'),
	RegexMatcher(r'[\(\)\[\]\{\}]', 0, 'bracket'),
	RegexMatcher(r'\s+', -1, 'whitespace'),
	RegexMatcher(r';', 0, 'statement'),
	RegexMatcher(r',', 0, 'comma'),
	RegexMatcher(r':', 0, 'colon'),
]

def tokens(code, matchers = matchers):
	return Lexer(matchers, code)

def tokenize(code, matchers = matchers):
	return list(tokens(code, matchers))

if __name__ == '__main__':
	for i in tokens(open(sys.argv[1], 'r').read()): print(i)
