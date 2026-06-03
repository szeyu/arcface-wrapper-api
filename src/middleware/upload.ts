import multer from "multer";
import path from "path";

// Configure multer to store files in memory as buffers
const storage = multer.memoryStorage();

// File filter to accept only images
const fileFilter = (
  req: Express.Request,
  file: Express.Multer.File,
  cb: multer.FileFilterCallback
) => {
  // Accept common image formats
  const allowedMimeTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/avif"];
  const allowedExtensions = [".jpg", ".jpeg", ".png", ".webp", ".avif"];
  const extension = path.extname(file.originalname).toLowerCase();

  if (
    allowedMimeTypes.includes(file.mimetype) ||
    (file.mimetype === "application/octet-stream" && allowedExtensions.includes(extension))
  ) {
    cb(null, true);
  } else {
    cb(new Error("Invalid file type. Only JPEG, PNG, WEBP, and AVIF images are allowed."));
  }
};

// Configure multer with memory storage and file filter
export const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});
