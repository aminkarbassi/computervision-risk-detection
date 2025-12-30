from icrawler.builtin import BingImageCrawler
from pathlib import Path


def download_images(query: str, output_dir: Path, max_images: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": str(output_dir)}
    )
    crawler.crawl(
        keyword=query,
        max_num=max_images
    )


if __name__ == "__main__":
    base_dir = Path("data/processed")

    # Training images
    download_images(
        query="aerial roof house",
        output_dir=base_dir / "train" / "roof",
        max_images=10
    )

    download_images(
        query="aerial garden forest",
        output_dir=base_dir / "train" / "not_roof",
        max_images=10
    )

    # Validation images
    download_images(
        query="satellite roof building",
        output_dir=base_dir / "val" / "roof",
        max_images=10
    )

    download_images(
        query="satellite forest park",
        output_dir=base_dir / "val" / "not_roof",
        max_images=10
    )

    print("Image download completed.")

