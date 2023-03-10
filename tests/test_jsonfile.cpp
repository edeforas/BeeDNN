#include <iostream>
using namespace std;

#include "JsonFile.h"

int main()
{
	JsonFile jsw;
	jsw.add("key1", 17);
	jsw.enter_section("subsection");
	jsw.add("key2", 24);
	jsw.write("test.json");
	
	JsonFile jsr;
	jsr.read("test.json");

	return 0;
}


