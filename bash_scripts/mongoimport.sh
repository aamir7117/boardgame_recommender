#!/bin/bash
for i in $( ls ); do
	mongoimport --db bgg_data --collection ${i%%.*} --file $i
	mongoimport --db bgg_data --collection master --file $i

done
