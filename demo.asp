
start(a).
road(a,b). road(b,c). road(c,d). road(d,a).

visit(Y) :- road(X,Y), start(X).
visit(Y) :- road(X,Y), visit(X).

