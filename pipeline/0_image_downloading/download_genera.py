from downloadGenusImages import downloadGenusImages

# download genus images in ~4.5 gb chunks downloaded every hour (in accordance with iNat limit)
# iterates through genera in genus list, first downloading records then downloading images using records
downloadGenusImages(start_index=1,end_index=3,skip_records=False,skip_images=False,pylib_root="pylib")