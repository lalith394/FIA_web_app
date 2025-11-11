"use client";

import * as React from "react";
import Link from "next/link";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu";
import { Moon, Sun, BookAIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

export default function NavigationBar() {
  const { theme, setTheme } = useTheme();
  const navItems = [
    {
      item: "Docs",
      reference: "/docs",
      icon: BookAIcon
    },
    {
      item: "Generate",
      reference: "/generate",
      icon: BookAIcon
    },
    {
      item: "About Us",
      reference: "/aboutUS",
      icon: BookAIcon
    }
  ]

  return (
    <nav className="flex items-center justify-between w-full px-6 py-4 border-b bg-background">
      {/* Left: App title */}
      <Link
        href="/"
        className="text-2xl font-semibold tracking-tight text-foreground hover:text-primary transition-colors"
      >
        FIA Web UI
      </Link>

      {/* Right: Navigation menu */}
      <div className="flex items-center space-x-6">
        <NavigationMenu>
          <NavigationMenuList>
            {navItems.map(({item, reference, icon: Icon}, index) => (
              <NavigationMenuItem key={index}>
                <NavigationMenuLink className="text-foreground hover:text-primary transition-colors flex flex-row justify-center items-center" href={reference}>
                <Icon/> {item}
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>

        {/* Dark/Light toggle */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="rounded-full cursor-pointer"
        >
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </div>
    </nav>
  );
}
