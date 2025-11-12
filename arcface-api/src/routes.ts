import express from "express";
import { storeEmbedding, compareEmbeddings, searchEmbeddings } from "./controllers/embeddingController";
import { listImages, getImage, deleteImage } from "./controllers/imageController";

const router = express.Router();

router.post("/store_embedding", storeEmbedding);
router.post("/compare", compareEmbeddings);
router.post("/search", searchEmbeddings);

router.get("/image/:id", getImage);
router.get("/list", listImages);
router.delete("/item/:id", deleteImage);

export default router;

