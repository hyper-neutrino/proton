import lexer, sys

from errors import *

class ASTNode:
	def __init__(self, token, children, **kwargs):
		types = token.type.split(':')
		self.explicit_type = len(types) > 1
		self.type = token.type if len(types) < 2 else types[1]
		self.token = token
		self.children = children
		self.values = kwargs
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

def match(left, right):
	return right == None or isinstance(right, tuple) and any(k in right for k in left) or isinstance(right, (type(print), type(lambda: 0))) and right(left) or right in left

class SingleASTMatcher:
	def __init__(self, type = None, tokentype = None, content = None, children = None):
		self.type = type
		self.tokentype = tokentype
		self.content = content
		self.children = children
	def match(self, ast):
		return match(ast.type.split('/'), self.type) and match(ast.token.type.split('/'), self.tokentype) and match([ast.token.content], self.content) and match([ast.children], self.children)
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
		self.conditions = [SingleASTMatcher(*condition) for condition in conditions]
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

class BracketExprMatcher:
	def __init__(self, open, close, expr_type, reiter = True):
		self.open = open
		self.close = close
		self.expr_type = expr_type
		self.reiter = reiter
		self.RTL = False

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
						if rule.reiter: result = Parser(self.rules, result, True).construct_tree()
						if hasattr(rule, 'postconfig'): result = rule.postconfig(result)
						if not isinstance(result, list): result = [result]
						self.tree[iterindex:iterindex + match[0]] = result
						iterindex = base
						modified = True
					else:
						iterindex += delta
					if iterindex >= len(self.tree) and rule.RTL: iterindex = len(self.tree) - 1
		return self.tree

def recurstr(array):
	if isinstance(array, list):
		return str(list(map(recurstr, array)))
	return str(array)

def FT(key):
	print('key: ' + recurstr(key))
	return key

matchers = [
	BracketMatcher(open, close, bracket_type) for open, close, bracket_type in [('(', ')', 'bracket'), ('[', ']', 'list'), ('{', '}', 'codeblock')]
] + [
	PatternMatcher([('expression',), ('unifix_operator',)], lambda x, y: y.addChild(x).addType('postfix/expression').rmType('unifix_operator')),
	PatternMatcher([('unifix_operator',), ('expression',)], lambda x, y: x.addChild(y).addType('prefix/expression').rmType('unifix_operator')),
] + [
	PatternMatcher([('expression',), ('bracket',)], lambda x, y: ASTNode(lexer.Token('call/expression', ''), [x] + (y.children[0].children if y.children and y.children[0].token.type == 'comma' else y.children))),
	PatternMatcher([('expression',), ('list',)], lambda x, y: ASTNode(lexer.Token('getitem/expression', ''), [x, ASTNode(lexer.Token('expression/comma_expr', ''), y.children[0].children) if y.children and y.children[0].token.type == 'comma' else y.children[0]])),
] + [
	PatternMatcher([('expression',), ('binary_RTL', 'binary_RTL', row), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression'), True) for row in lexer.binary_RTL
] + [
	PatternMatcher([('expression',), ('binary_operator', 'binary_operator', row), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('expression').rmType('binary_operator')) for row in lexer.binary_operators
] + [
	PatternMatcher([(('expression', 'comma_expr'),), ('comma',), ('expression',)], lambda x, y, z: y.addChild(x).addChild(z).addType('comma_expr') if 'expression' in x.type.split('/') else x.addChild(z)),
	PatternMatcher([('expression',), ('comma',)], lambda x, y: y.addChild(x).addType('comma_expr'))
] + [
	# PatternMatcher([('open_slice',), ('colon',)], lambda x, y: x.addChild(ASTNode(lexer.Token('none', '')))),
	# PatternMatcher([])
] + [
	PatternMatcher([(lambda k: 'expression' in k and 'statement' not in k,), ('statement', 'statement', 'semicolon')], lambda x, y: x)
] + [
	MultiTypeMatch([
		PatternMatcher([('expression',), ('keyword', 'keyword', 'unless'), ('expression',), (('expression', 'statement'),)], lambda w, x, y, z: x.addChildren([w, y, z]).addType('statement')),
		PatternMatcher([('keyword', 'keyword', 'if'), ('expression',), (('expression', 'statement'),)], lambda x, y, z: x.addChild(y).addChild(z).addType('statement')),
		PatternMatcher([('keyword', 'keyword', 'else'), (('expression', 'statement'),)], lambda x, y: x.addChild(y).addType('statement')),
		PatternMatcher([('keyword', 'keyword', 'if'), ('keyword', 'keyword', 'else')], lambda x, y: x.addChild(y.children[0]))
	], RTL = True)
]

def parse(tokens):
	return Parser(matchers, tokens).construct_tree()
