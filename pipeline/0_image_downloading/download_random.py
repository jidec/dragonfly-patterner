from downloadRandomSingleImages import downloadRandomSingleImages

# download random single images
# does not work in batch (i.e. more than 500) as making a large number of these API calls at once causes a temp ban
downloadRandomSingleImages(n=5,pylib_root="pylib")