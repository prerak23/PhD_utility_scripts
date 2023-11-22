#include <stdio.h>
#include <time.h>
#include "dm.h"


void getSeconds(unsigned long *par) {
   /* get the current number of seconds */
   *par = time( NULL );
   return;
}
