all:
	gcc main.c glad_gl.c -Ofast -lglfw -lm -o porydrive

install:
	cp porydrive $(DESTDIR)

uninstall:
	rm $(DESTDIR)/porydrive

clean:
	rm porydrive