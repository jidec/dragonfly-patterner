macro "SIOUX Segment Helper" {

first = true; 

part_name = getString("Enter a segment part name - leave empty if not annotating multiple parts:", "");
if(part_name != ""){
	part_name = part_name + "-";
}

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
			
			params = "blocksize=100 histogram=256 maximum_slope=7 mask=*None* fast_(less_accurate)";
			run("Enhance Local Contrast (CLAHE)",params);
			
			run("SIOX: Simple Interactive Object Extraction");
			setTool("brush");

			waitForUser("WaitForUser", "Click OK after you hit *Create mask* in the segmenter and finished editing the B/W segment if not scrapping it");
			message = "Click YES to save finished mask and move To next image, NO to skip this image if the segmenter fails badly and is not worth fixing";
			bool = getBoolean(message, "YES: Save mask", "NO: Skip image");
			
			if(bool){
				output = replace(input+file, ".jpg", "");
				output = output + "_" + part_name + "mask.jpg";
				saveAs("/formats/jpg", output);
				close();
				close();
			}
			if(!bool){
				run("Subtract...", "value=255");
				output = replace(input+file, ".jpg", "");
				output = output + "_" + part_name + "mask.jpg";
				saveAs("/formats/jpg", output);
				close();
				close();
			}
		}
	}
}