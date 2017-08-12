import lexer, sys

from errors import *
from utils import *

class ASTNode:
	def __init__(self, token, children, **kwargs):
		types = token.type.split(':')
		self.explicit_type = len(types) > 1
		self.type = token.type if len(types) < 2 else types[1]
		self.token = token
		self.children = children
		self.values = kwargs
	def setTokenType(self, type):
		self.token.type = type
		return self
	def setTokenContent(self, content):
		self.token.content = content
		return self
	def addChild(self, child):
		self.children.append(child)
		return self
	def addChildren(self, children):
		for child in children:
			self.addChild(child)
		return self
	def removeChild(self, child):
		if child in self.children:
			self.children.remove(child)
		return self
	def removeChildren(self, children):
		for child in children:
			self.removeChild(child)
		return self
	def setType(self, type):
		self.type = type
		return self
	def addType(self, type):
		self.type = self.type.rstrip('/') + '/' + type
		return self
	def rmType(self, type):
		self.type = '/'.join(t for t in self.type.split('/') if t != type)
		return self
	def setTypeIfNone(self, type):
		self.explicit_type ^= True
		if self.explicit_type:
			self.setType(type)
		return self
	def __str__(self, head = True):
		string = head * 'AST:' + str(self.token) + ' (%s)' % self.type
		for child in self.children:
			string += '\n' + '\n'.join('|\t' + line for line in (child.__str__(False) if isinstance(child, ASTNode) else 'FOREIGN: ' + str(child)).split('\n'))
		return string
	def __eq__(self, other):
		return isinstance(other, ASTNode) and other.type == self.type and other.token == self.token and other.children == self.children

def isiter(obj):
	return hasattr(obj, '__iter__') and not isinstance(obj, str)

def match(left, right):
	return right == None or isinstance(right, tuple) and any(k in right for k in left) or isinstance(right, (type(print), type(lambda: 0))) and right(left) or any(right == elem for elem in left)

class SingleASTMatcher:
	def __init__(self, type = None, tokentype = None, content = None, children = None):
		self.type = type
		self.tokentype = tokentype
		self.content = content
		self.children = children
	def match(self, ast):
		return match(ast.type.split('/'), self.type) and match(ast.token.type.replace(':', '/').split('/'), self.tokentype) and match([ast.token.content], self.content) and match([ast.children], self.children)
	def __str__(self):
		return '<SingleASTMatcher type=%s tokentype=%s content=%s children=%s>' % (self.type, self.tokentype, self.content, self.children)
	def __repr__(self):
		return str(self)

class ParserMatcher:
	def __init__(self, matcher, getter, RTL = False, reiter = False, skip = lambda *a: 0):
		self.matcher = matcher
		self.getter = getter
		self.RTL = RTL
		self.reiter = reiter
		self.skip = skip
	def match(self, nodes):
		return self.matcher(nodes)
	def get(self, nodes, match):
		return self.getter(nodes, match)
	def skip(self, nodes, match):
		return self.skip(nodes, match)

class PatternMatcher:
	def __init__(self, conditions, shaper, RTL = False, reiter = False):
		self.conditions = [condition if isinstance(condition, (SingleASTMatcher, PatternMatcher, ParserMatcher, BracketMatcher)) else SingleASTMatcher(*condition) for condition in conditions]
		self.shaper = shaper
		self.reiter = reiter
		self.RTL = RTL
	def match(self, nodes):
		return (len(self.conditions), nodes[:len(self.conditions)]) if len(nodes) >= len(self.conditions) and all(matcher.match(node) for matcher, node in zip(self.conditions, nodes)) else None
	def get(self, nodes, match):
		result = self.shaper(*match)
		if not isinstance(result, list): result = [result]
		return result
	def skip(self, nodes, match):
		return 0

class MultiTypeMatch:
	def __init__(self, matchers, RTL = False, reiter = False):
		self.matchers = matchers
		self.reiter = reiter
		self.RTL = RTL
	def match(self, nodes):
		for matcher in self.matchers:
			match = matcher.match(nodes)
			if match:
				return (match[0], (matcher, match[1]))
	def get(self, nodes, match):
		return match[0].get(nodes, match[1])
	def skip(self, nodes, match):
		return match[0].skip(nodes, match[1])

class BracketMatcher:
	def __init__(self, open, close, bracket_type, reiter = True):
		self.open = open
		self.close = close
		self.bracket_type = bracket_type
		self.reiter = reiter
		self.RTL = False
	def match(self, nodes):
		if nodes and 'bracket' in nodes[0].token.type.split('/'):
			if nodes[0].token.content == self.open:
				bracket = 1
				index = 1
				while index < len(nodes):
					if 'bracket' in nodes[index].token.type.split('/'):
						if nodes[index].token.content == self.open:
							bracket += 1
						elif nodes[index].token.content == self.close:
							bracket -= 1
						if bracket == 0: return (index + 1, nodes[1:index])
					index += 1
				if bracket: raise UnclosedBracketError()
			elif nodes[0].token.content == self.close:
				raise UnopenedBracketError()
		return None
	def get(self, nodes, match):
		return match
	def postconfig(self, trees):
		return ASTNode(lexer.Token('expression/bracket_expr/' + self.bracket_type, ''), trees[:])
	def skip(self, nodes, match):
		return 0

class Parser:
	def __init__(self, rules, tokens, config = False):
		self.rules = rules
		self.tree = tokens if config else [ASTNode(token, []) for token in tokens]
	def construct_tree(self):
		rules = self.rules[:]
		index = 0
		modified = False
		while rules:
			if index % len(rules) == 0 and index:
				modified ^= True
				if modified:
					break
			rule = rules[index % len(rules)]
			index += 1
			tree = None
			while tree != self.tree:
				tree = self.tree[:]
				iterindex = base = len(self.tree) - 1 if rule.RTL else 0
				delta = [1, -1][rule.RTL]
				while (rule.RTL and iterindex >= 0) or (not rule.RTL and iterindex < len(self.tree)):
					match = rule.match(self.tree[iterindex:])
					if match:
						result = rule.get(self.tree[iterindex:], match[1])
						if rule.reiter:
							if rule.reiter is True:
								result = Parser(self.rules, result, True).construct_tree()
							else:
								result = Parser(self.rules + rule.reiter, result, True).construct_tree()
						if hasattr(rule, 'postconfig'): result = rule.postconfig(result)
						if not isinstance(result, list): result = [result]
						self.tree[iterindex:iterindex + match[0]] = result
						iterindex = base
						modified = True
					else:
						iterindex += delta
					if iterindex >= len(self.tree) and rule.RTL: iterindex = len(self.tree) - 1
			if modified:
				modified = False
				index = 0
		return self.tree

special = ['=', '.']

matchers = [
	PatternMatcher([('bracket', 'bracket', '('), (('binary_operator', 'binary_RTL'),), ('bracket', 'bracket', ')')], lambda x, y, z: x.addChild(y).addType('opfunc/expression/bracket_expr').rmType('bracket').setTokenType('').setTokenContent(''))
] + [
	MultiTypeMatch([
		PatternMatcher([('expression', lambda x: 'literal' not in x), (lambda x: 'bracket_expr' in x and 'bracket' in x,)], lambda x, y: ASTNode(lexer.Token('call/expression', ''), [x] + (y.children[0].children if y.children and y.children[0].token.type == 'comma' else y.children))),
		PatternMatcher([('expression',), ('list',)], lambda x, y: ASTNode(lexer.Token('getitem/expression', ''), [x, ASTNode(lexer.Token('expression/comma_expr', ''), y.children[0].children) if y.children and y.children[0].token.type == 'comma' else y.children[0]])),
		PatternMatcher([('expression',), ('binary_operator', 'binary_operator', '.'), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('binary_operator'))
	])
] + [
	BracketMatcher(open, close, bracket_type) for open, close, bracket_type in [('(', ')', 'bracket'), ('[', ']', 'list'), ('{', '}', 'codeblock')]
] + [
	PatternMatcher([('bracket', 'bracket', '', lambda k: any(x.token.type == 'colon' for x in k[0]))], lambda x: x.addChild(x.children[0]).addChild(x.children[2]).removeChildren(x.children[:3]).setType('foriter')),
	PatternMatcher([('bracket', 'bracket', '', lambda k: any(x.token.type == 'arrow' for x in k[0]))], lambda x: x.addChild(x.children[0]).addChild(x.children[2]).removeChildren(x.children[:3]).setType('whilearrow')),
] + [
	PatternMatcher([('expression',), ('unifix_operator',)], lambda x, y: y.addChild(x).addType('postfix/expression').rmType('unifix_operator')),
	PatternMatcher([('unifix_operator',), ('expression',)], lambda x, y: x.addChild(y).addType('prefix/expression').rmType('unifix_operator')),
] + [
	PatternMatcher([('expression',), ('binary_RTL', 'binary_RTL', tuple(elem for elem in row if elem not in special)), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('binary_RTL'), True) for row in lexer.binary_RTL
] + [
	PatternMatcher([('expression',), ('binary_operator', 'binary_operator', tuple(elem for elem in row if elem not in special)), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('binary_operator')) for row in lexer.binary_operators
] + [
	PatternMatcher([('expression',), ('keyword', 'keyword', name), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('keyword')) for name in ['from', 'as']
] + [
	PatternMatcher([(('expression', 'comma_expr'),), ('comma',), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('comma_expr/expression') if 'comma_expr' not in x.type.split('/') else x.addChild(z)),
	PatternMatcher([('expression',), ('comma',)], lambda x, y: y.addChild(x).addType('comma_expr/expression'))
] + [
	PatternMatcher([('expression',), ('binary_RTL', 'binary_RTL', '='), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('binary_RTL'), True)
] + [
	PatternMatcher([(lambda k: 'expression' in k and 'statement' not in k,), ('statement', 'statement', ';')], lambda x, y: x.addType('statement'))
] + [
	PatternMatcher([(('expression'),), ('keyword', 'keyword', 'exist')], lambda x, y: y.addChild(x).addType('expression/exist').rmType('keyword')),
	PatternMatcher([(('expression'),), ('keyword', 'keyword', 'exists')], lambda x, y: y.addChild(x).addType('expression/exist').rmType('keyword')),
	PatternMatcher([(('expression'),), ('keyword', 'keyword', 'exist not')], lambda x, y: y.addChild(x).addType('expression/notexist').rmType('keyword')),
	PatternMatcher([(('expression'),), ('keyword', 'keyword', 'exists not')], lambda x, y: y.addChild(x).addType('expression/notexist').rmType('keyword')),
] + [
	PatternMatcher([('keyword', 'keyword', 'import'), ('expression',)], lambda x, y: x.addChild(y).addType('expression').rmType('keyword'))
] + [
	PatternMatcher([('keyword', 'keyword', statement)], lambda x: x.setType(statement + '_statement/expression')) for statement in ['break', 'continue']
] + [
	MultiTypeMatch([
		PatternMatcher([('keyword', 'keyword', 'for'), ('expression',), ('colon',), ('expression',), (('expression', 'statement'),)], lambda v, w, x, y, z: v.addChildren([w, y, z]).addType('statement/foreach').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'for')] + [(('expression', 'statement'),)] * 4, lambda v, w, x, y, z: v.addChildren([w, x, y, z]).addType('statement').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'for'), ('foriter',), (('expression', 'statement'),)], lambda x, y, z: x.addChildren(y.children).addChild(z).addType('statement/foreach').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'for'), ('bracket',), (('expression', 'statement'),)], lambda x, y, z: x.addChildren(y.children).addChild(z).addType('statement').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'while'), ('expression',), ('arrow',), ('expression',), (('expression', 'statement'),)], lambda v, w, x, y, z: v.addChildren([w, y, z]).addType('statement/whileas').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'while'), ('expression',), (('expression', 'statement'),)], lambda x, y, z: x.addChild(y).addChild(z).addType('statement').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'while'), ('whilearrow',), (('expression', 'statement'),)], lambda x, y, z: x.addChildren(y.children).addChild(z).addType('statement/whileas').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'while'), ('bracket',), (('expression', 'statement'),)], lambda x, y, z: x.addChildren(y.children).addChild(z).addType('statement'.rmType('keyword'))),
		PatternMatcher([('expression',), ('keyword', 'keyword', 'unless'), ('expression',), (('expression', 'statement'),)], lambda w, x, y, z: x.addChildren([w, y, z]).addType('statement').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'if'), ('expression',), (('expression', 'statement'),)], lambda x, y, z: x.addChild(y).addChild(z).addType('statement').rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'else'), (('expression', 'statement'),)], lambda x, y: x.addChild(y).addType('statement').rmType('keyword')),
		PatternMatcher([('statement', 'keyword', 'if'), ('statement', 'keyword', 'else')], lambda x, y: x.addChild(y.children[0]).rmType('keyword')),
		PatternMatcher([('keyword', 'keyword', 'try'), (('expression', 'statement'),), ('keyword', 'keyword', 'except'), ('expression',), (('expression', 'statement'),)], lambda v, w, x, y, z: v.addChildren([w, y, z]).addType('statement').rmType('keyword')),
	], RTL = True)
] + [
	PatternMatcher([('expression',), ('ternary',), ('expression',), ('colon',), ('expression',)], lambda v, w, x, y, z: w.addChildren([v, x, z]).addType('expression').rmType('ternary'))
]

def parse(tokens):
	return Parser(matchers, tokens).construct_tree()

if __name__ == '__main__':
	for i in parse(lexer.tokenize(open(sys.argv[1], 'r').read())): print(i)
