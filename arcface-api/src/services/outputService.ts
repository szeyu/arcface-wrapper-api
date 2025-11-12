import { promises as fs } from "fs";
import path from "path";

const OUTPUT_DIR = path.join(process.cwd(), "output");

export const saveImageToOutput = async (id: string, imageBuffer: Buffer): Promise<string> => {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  let extension = "jpeg"; // default
  const magicBytes = imageBuffer.subarray(0, 4);

  // PNG: 89 50 4E 47
  if (magicBytes[0] === 0x89 && magicBytes[1] === 0x50 && magicBytes[2] === 0x4E && magicBytes[3] === 0x47) {
    extension = "png";
  }
  // JPEG: FF D8 FF
  else if (magicBytes[0] === 0xff && magicBytes[1] === 0xd8 && magicBytes[2] === 0xff) {
    extension = "jpeg";
  }
  // GIF: 47 49 46 38
  else if (magicBytes[0] === 0x47 && magicBytes[1] === 0x49 && magicBytes[2] === 0x46 && magicBytes[3] === 0x38) {
    extension = "gif";
  }
  // WebP: Check for RIFF...WEBP
  else if (
    imageBuffer.length >= 12 &&
    magicBytes[0] === 0x52 &&
    magicBytes[1] === 0x49 &&
    magicBytes[2] === 0x46 &&
    magicBytes[3] === 0x46
  ) {
    const webpHeader = imageBuffer.subarray(8, 12);
    if (webpHeader[0] === 0x57 && webpHeader[1] === 0x45 && webpHeader[2] === 0x42 && webpHeader[3] === 0x50) {
      extension = "webp";
    }
  }

  const outputPath = path.join(OUTPUT_DIR, `${id}.${extension}`);
  await fs.writeFile(outputPath, imageBuffer);
  return outputPath;
};


