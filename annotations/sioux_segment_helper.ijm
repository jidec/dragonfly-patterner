macro "SIOUX Segment Helper" {

first = true; 

input = getDirectory(""); // images

processFolder(input);

function processFolder(input) {
	list = getFileList(input);

	// for file in list
	for (i = 0; i < list.length; i++) {
		
		file = list[i]; 
		
		// check if is a Mask 
		isMask = false;
		if(endsWith(file, "_mask.jpg")){
			isMask = true;
		}
		
		// check every other file in list to see if has a Mask
		hasMask = false;
		name = replace(file, ".jpg", "");
		for (j = 0; j < list.length; j++) {
			file2 = list[j];
			if(endsWith(file2, name + "_mask.jpg")){
				hasMask = true;
			}
		}
		
		// send to SIOUX if image is not a mask and mask doesn't exist yet 
		if(!isMask && !hasMask){
			open(input + file);
			run("SIOX: Simple Interactive Object Extraction");
			setTool("brush");
			title = "WaitForUser";
			msg = "Click OK to Save Mask and Move To Next Image";
			waitForUser(title, msg);
			output = replace(input+file, ".jpg", "");
			output = output + "_mask.jpg";
			saveAs("/formats/jpg", output);
			close();
			close();
		}
	}
}