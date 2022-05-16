from downloadiNatGenusImages import downloadiNatGenusImages

# download genus images in ~4.5 gb chunks downloaded every hour (in accordance with iNat limit)
# iterates through genera in genus list, first downloading records then downloading images using records
downloadiNatGenusImages(start_index=41,end_index=42,skip_records=False,skip_images=False)