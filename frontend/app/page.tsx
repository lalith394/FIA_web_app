import HeroSection from "./components/hero";
import NavigationBar from "./components/navigation";

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto">
    <NavigationBar />
    <HeroSection />
    </div>
  );
}
