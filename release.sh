clang main.c glad_gl.c -Ofast -lglfw -lm -o porydrive
i686-w64-mingw32-gcc main.c glad_gl.c -Ofast -Llib -lglfw3dll -lm -o porydrive.exe
upx porydrive
upx porydrive.exe
cp porydrive porydrive.AppDir/usr/bin/
./appimagetool-x86_64.AppImage porydrive.AppDir