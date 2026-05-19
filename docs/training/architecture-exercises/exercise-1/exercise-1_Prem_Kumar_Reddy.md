# Exercise 1 – Library Book Borrowing

| | |
|---|---|
| **Author:** | Prem Kumar Reddy Kothakapu |
| **Date:** | 11-May-2026 |
| **Exercise:** | Architecture Modelling with ArchiMate |
| **System:** | Borrowing a physical book from a public library using a library card. |

## System overview

The system supports a **Library Member** (patron) and a **Librarian** to complete the full book borrowing and returning cycle. The core business services are **Book Lending Service** and **Book Return Service**. The main business processes include: verifying the library card, searching for a book, issuing the book, returning it, and updating inventory. The library uses a library card and a barcode scanner to identify books and members. The model spans all three layers and includes all required relationship types.

## Business layer

The **Library Member** is assigned to the *Book Lending Service* and to the *Return Book Process*. The **Librarian** is assigned to the *Issue Book Process* and to the *Verify Library Card Process*. The *Book Lending Service* is realised by three processes: *Verify Library Card Process*, *Issue Book Process*, and *Update Inventory Process* (through a triggering link). The *Book Return Service* is realised by the *Return Book Process*.

Key business objects: **Book** (read by the issue process), **Library Card** (used in verification), **Borrow Request** (created during issue), and **Loan Transaction** (updated by issue and return). The *Issue Book Process* has **readwrite** access to *Loan Transaction* and **read** access to *Book*. The *Return Book Process* also has **readwrite** access to *Loan Transaction*.

## Application layer

The **Library Management System** (application component) provides the **Borrowing Management Service**, which serves the *Issue Book Process*. The **Library Catalogue System** provides the **Book Search Service**, which serves the *Search Book Process*.

The management system has **readwrite** access to the **Member Record** (data object). The catalogue system has **readwrite** access to **Book Inventory Data**. A third data object, **Loan Transaction Data**, is also present and linked to the business object *Loan Transaction*.

## Technology layer

The **Library Server** (node) hosts both application components (assignment relationships). It also **composes** the **Barcode Scanner** (device). The scanner sends a **flow** of data (scanned book or card information) to the server.

The **Database Hosting Service** and the **Network Service** (technology services) both **serve** the Library Server, enabling communication with external systems and data storage.

## Relationships used

- **Assignment**: Library Member → Book Lending Service; Library Member → Return Book Process; Librarian → Issue Book Process; Librarian → Verify Library Card Process; Library Server → Library Management System; Library Server → Library Catalogue System.
- **Realization**: Issue Book Process → Book Lending Service; Return Book Process → Book Return Service; Verify Library Card Process → Book Lending Service.
- **Serving**: Book Search Service → Search Book Process; Borrowing Management Service → Issue Book Process; Library Management System → Borrowing Management Service; Library Catalogue System → Book Search Service; Database Hosting Service → Library Server; Network Service → Library Server.
- **Composition**: Library Server → Barcode Scanner.
- **Flow**: Barcode Scanner → Library Server.
- **Access** (with types): Issue Book Process → Loan Transaction (readwrite); Issue Book Process → Book (read); Return Book Process → Loan Transaction (readwrite); Library Management System → Member Record (readwrite); Library Catalogue System → Book Inventory Data (readwrite).

All required relationship types (Assignment, Realization, Serving, Composition, Flow) are present. The model shows a complete flow from the user’s request down to the physical scanner and network services, with clear cross‑layer links.