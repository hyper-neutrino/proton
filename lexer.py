import re, ast, sys, sympy

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
	def __init__(self, regex, group, tokentype, processor = lambda x: x, offset = 0, **kwargs):
		self.regex = regex
		self.group = group
		self.tokentype = tokentype
		self.processor = processor
		self.offset = offset
		self.values = kwargs
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
					if token is not None:
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
	('in', 'not in', 'is', 'are', 'is not', 'are not'),
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

keywords = ['if', 'else', 'unless', 'while', 'for', 'try', 'except', 'exist not', 'exist', 'exists not', 'exists', 'break', 'continue', 'import', 'include', 'as', 'from', 'to', 'by', 'timeof']

ignore = ('not',)

operators = sum(binary_operators, ()) + sum(binary_RTL, ()) + tuple(unifix_operators)

matchers = [
	RegexMatcher(r'#.+', -1, 'comment'),
	RegexMatcher(r'/\*([^*]|\*[^/])*\*/', -1, 'comment'),
	RegexMatcher(r'\d*\.\d+j', 0, 'literal:expression', complex),
	RegexMatcher(r'\d+j', 0, 'literal:expression', complex),
	RegexMatcher(r'\d+\s*/\s*\d+', 0, 'literal:expression', sympy.Rational),
	RegexMatcher(r'\d*\.\d+', 0, 'literal:expression', float),
	RegexMatcher(r'\d+', 0, 'literal:expression', int),
	RegexMatcher(r'"([^"\\]|\\.)*"', 0, 'literal:expression', lambda x: ast.literal_eval('""%s""' % x)),
	RegexMatcher(r"'([^'\\]|\\.)*'", 0, 'literal:expression', lambda x: ast.literal_eval("''%s''" % x)),
	ErrorMatcher(RegexMatcher(r'"([^"\\]|\\.)*', 0, ''), UnclosedStringError),
	ErrorMatcher(RegexMatcher(r"'([^'\\]|\\.)*", 0, ''), UnclosedStringError),
	RegexMatcher('(%s)' % '|'.join(['(%s)[^A-Za-z_]' % keyword for keyword in keywords]), 1, 'keyword', lambda x: x[:-1], -1),
	LexerMatcher(lambda code: re.match('[A-Za-z_][A-Za-z_0-9]*', code), lambda code, match: None if match.group() in operators + ignore else (match.end(), Token('keyword' if match.group() in keywords else 'identifier:expression', match.group()))),
	RegexMatcher(r';', 0, 'statement'),
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
