import lexer, parser, interpreter, errors, sys, traceback

if __name__ == '__main__':
	if sys.argv[1:]:
		interpreter.complete(parser.parse(lexer.tokenize(open(sys.argv[1], 'r').read())))
	else:
		code = ''
		while True:
			code += input('... ' if code else '>>> ') + '\n'
			try:
				interpreter.complete(parser.parse(lexer.tokenize(code)))
				code = ''
			except errors.UnclosedBracketError:
				pass
			except:
				print(traceback.format_exc())
				code = ''
