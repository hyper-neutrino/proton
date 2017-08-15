import re, ast, sys

from errors import *
from utils import *

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
	def __init__(self, matcher, getter, getlast = False, skip = lambda *a: 0):
		self.matcher = matcher
		self.getter = getter
		self.skip = skip
		self.getlast = getlast
	def match(self, *code_or_last):
		return self.matcher(*code_or_last)
	def get(self, code, match):
		return self.getter(code, match)
	def skip(self, code, match):
		return self.skip(code, match)

class RegexMatcher(LexerMatcher):
	def __init__(self, regex, group, tokentype, processor = lambda x: x, offset = 0, **kwargs):
		self.regex = regex
		self.group = group
		self.tokentype = tokentype
		self.processor = processor
		self.offset = offset
		self.values = kwargs
		self.getlast = False
	def match(self, code):
		return re.match(self.regex, code)
	def get(self, code, match):
		return (match.span()[1] + self.offset, Token(self.tokentype, self.processor(match.group(self.group)), **self.values))
	def skip(self, code, match):
		return match.span()[1] * (self.group == -1)

class ErrorMatcher(LexerMatcher):
	def __init__(self, matcher, errortype):
		self.matcher = matcher
		self.errortype = errortype
		self.getlast = False
	def match(self, code):
		return self.matcher.match(code)
	def get(self, code, match):
		raise self.errortype()
	def skip(self, code, match):
		return self.matcher.skip(code, match)

identifier_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'

def findv(operator, values):
	for value in values:
		if operator in value[1]: return value[0]

def oper_matcher(array, values, counter = []):
	return LexerMatcher(lambda code: (lambda x: (lambda y: y if y not in counter else '')(x and max(x, key=len)))([operator for operator in array if code.startswith(operator)]), lambda code, match: (len(match), Token(findv(match, values), match)))

class Lexer:
	def __init__(self, rules, code):
		self.rules = rules
		self.code = code
		self.index = 0
		self.last = None
	def __iter__(self):
		return self
	def __next__(self):
		if self.index >= len(self.code): raise StopIteration
		for rule in self.rules:
			code = self.code[self.index:]
			match = rule.match(code, self.last) if rule.getlast else rule.match(code)
			if match:
				skip = rule.skip(code, match)
				if skip:
					self.index += skip
					self.last = self.__next__()
					return self.last
				else:
					token = rule.get(code, match)
					if token is not None:
						self.index += token[0]
						self.last = token[1]
						return self.last
		raise RuntimeError('Unknown token at index %d: "...%s..."' % (self.index, self.code[self.index:][:10].replace('\n', '\\n')))

binary_RTL = [
	('**',),
	('=',)
]

binary_operators = [
	('.',),
	('**',),
	('>>', '<<'),
	('*', '/', '//'),
	('%',),
	('+', '-'),
	('>', '<', '<=', '>='),
	('&',),
	('|',),
	('^',),
	('**=', '*=', '/=', '//=', '+=', '-=', '>>=', '<<=', '%=', '&=', '|=', '&&=', '||='),
	('==', '!=', ':=', '=', '=:'),
	('&&', 'and'),
	('||', 'or'),
	('in', 'not in', 'is', 'are', 'is not', 'are not', 'inside', 'not inside'),
]

prefix_operators = ['!', '++', '--', '~', '@', '$', '$$']
postfix_operators = ['!', '++', '--']

unifix_operators = prefix_operators + postfix_operators

def recurstr(array):
	if isinstance(array, (map,)):
		array = list(array)
	if isinstance(array, list):
		return str(list(map(recurstr, array)))
	return str(array)

keywords = ['if', 'else', 'unless', 'while', 'for', 'try', 'except', 'exist not', 'exist', 'exists not', 'exists', 'break', 'continue', 'import', 'include', 'as', 'from', 'to', 'by', 'timeof', 'return']

ignore = ('not',)

operators = sum(binary_operators, ()) + sum(binary_RTL, ()) + tuple(unifix_operators)

def flags(key):
	flag = 0
	if 'a' in key:
		flag += re.ASCII
	if 'i' in key:
		flag += re.IGNORECASE
	if 'l' in key:
		flag += re.LOCALE
	if 'm' in key:
		flag += re.MULTILINE
	if 's' in key:
		flag += re.DOTALL
	if 'x' in key:
		flag += re.VERBOSE
	return flag

def intify(base):
	def inner(string):
		left, right = re.split('[^0-9]', string, 1)
		return int(right, base) * (base ** int(left))
	return inner

matchers = [
	RegexMatcher(r'#.+', -1, 'comment'),
	RegexMatcher(r'/\*([^*]|\*[^/])*\*/', -1, 'comment'),
	LexerMatcher(lambda code, last: None if last and 'operator' in last.type else re.match(r'/(([^/\\]|\\.)*)/([ailmsx]*)', code), lambda code, match: (match.end(), Token('literal:expression', re.compile(match.group(1), flags(match.groups()[-1])))), getlast = True),
	LexerMatcher(lambda code, last: None if last and 'operator' in last.type else re.match(r'/(([^/\\]|\\.)*)/', code), lambda code, match: (match.end(), Token('literal:expression', re.compile(match.group(1)))), getlast = True),
	RegexMatcher(r'\d+b[01]+', 0, 'literal:expression', intify(2)),
	RegexMatcher(r'\d+o[0-7]+', 0, 'literal:expression', intify(8)),
	RegexMatcher(r'\d+x[0-9a-fA-F]+', 0, 'literal:expression', intify(16)),
	RegexMatcher(r'\d+e\d+', 0, 'literal:expression', lambda x: (lambda y: int(y[0]) * 10 ** int(y[1]))(x.split('e'))),
	RegexMatcher(r'\d*\.\d+j', 0, 'literal:expression', complex),
	RegexMatcher(r'\d+j', 0, 'literal:expression', complex),
	RegexMatcher(r'\d*\.\d+', 0, 'literal:expression', float),
	RegexMatcher(r'\d+', 0, 'literal:expression', int),
	RegexMatcher(r'"([^"\\]|\\.)*"', 0, 'literal:expression', lambda x: ast.literal_eval('""%s""' % x)),
	RegexMatcher(r"'([^'\\]|\\.)*'", 0, 'literal:expression', lambda x: ast.literal_eval("''%s''" % x)),
	ErrorMatcher(RegexMatcher(r'"([^"\\]|\\.)*', 0, ''), UnclosedStringError),
	ErrorMatcher(RegexMatcher(r"'([^'\\]|\\.)*", 0, ''), UnclosedStringError),
	RegexMatcher('(%s)' % '|'.join(['(%s)[^A-Za-z_]' % keyword for keyword in keywords]), 1, 'keyword', lambda x: x[:-1], -1),
	LexerMatcher(lambda code: re.match('[A-Za-z_][A-Za-z_0-9]*', code), lambda code, match: None if match.group() in operators + ignore else (match.end(), Token('keyword' if match.group() in keywords else 'identifier:expression', match.group()))),
	RegexMatcher(r';', 0, 'semicolon'),
	RegexMatcher(r',', 0, 'comma'),
	RegexMatcher(r'\?', 0, 'ternary'),
	RegexMatcher(r':>', 0, 'maparrow'),
	RegexMatcher(r'->', 0, 'arrow'),
	RegexMatcher(r'=>', 0, 'lambda'),
	oper_matcher(operators, [('unifix_operator', unifix_operators), ('binary_RTL', sum(binary_RTL, ())), ('binary_operator', sum(binary_operators, ()))]),
	RegexMatcher(r':', 0, 'colon'),
	RegexMatcher(r'[\(\)\[\]\{\}]', 0, 'bracket'),
	RegexMatcher(r'\s+', -1, 'whitespace'),
]

def tokens(code, matchers = matchers):
	return Lexer(matchers, code)

def tokenize(code, matchers = matchers):
	return list(tokens(code, matchers))

if __name__ == '__main__':
	for i in tokens(open(sys.argv[1], 'r').read()): print(i)
