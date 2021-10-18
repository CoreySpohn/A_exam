for PHOTO in ./rv_example/*.{png}
   do
       BASE=$(basename $PHOTO)
    convert "$PHOTO" -quality 50% "./rv_example/compressed/${BASE%.*}.jpg"
   done
