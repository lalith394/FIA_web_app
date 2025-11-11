"use client";

import React from "react";
import Autoplay from "embla-carousel-autoplay";
import useEmblaCarousel from "embla-carousel-react";
import Image from "next/image";
import Link from "next/link";

export default function HeroSection() {
  const [emblaRef] = useEmblaCarousel({ loop: true }, [Autoplay({ delay: 3000 })]);

  const images = [
    "/images/1.png",
    "/images/2.png",
    "/images/3.png",
    "/images/4.png",
  ];

  return (
    <section className="flex flex-col md:flex-row items-center justify-between px-8 py-20 bg-background text-foreground">
      {/* Left Content */}
      <div className="flex-1 space-y-6">
        <h1 className="text-4xl md:text-5xl font-bold leading-tight tracking-tight">
          Building <span className="text-primary">compact</span> and <span className="text-primary">explainable</span> <br />
          <span className="text-primary">Deep Learning Models</span>
        </h1>
        <p className="text-lg text-muted-foreground">
          For Fundus Image Analysis
        </p>

        <div className="pt-4">
          <Link href={"/docs"}>
            <button className="px-6 py-3 bg-primary text-primary-foreground rounded-xl hover:scale-105 transition-transform cursor-pointer">
              Get Started
            </button>
          </Link>
        </div>
      </div>

      {/* Right Content: Carousel */}
      <div className="flex-1 mt-10 md:mt-0 md:ml-12 w-full max-w-lg">
        <div className="overflow-hidden rounded-2xl shadow-lg" ref={emblaRef}>
          <div className="flex">
            {images.map((src, index) => (
              <div key={index} className="flex-[0_0_100%] min-w-0">
                <Image
                  width={500}
                  height={500}
                  src={src}
                  alt={`Fundus Image ${index + 1}`}
                  className="w-full h-[350px] object-cover"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
